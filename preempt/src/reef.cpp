#include <list>
#include <mutex>
#include <memory>
#include <unordered_map>

#include "client.h"
#include "reef/reef.h"
#include "reef/utils.h"
#include "reef/stream.h"
#include "reef/command.h"
#include "reef/reef_hook.h"
#include "hook/cuda_hook.h"
#include "instrument/instrument.h"

using RFStream = reef::REEFStream;
using RFStreamPtr = std::shared_ptr<RFStream>;

using RFEvent = reef::EventRecordCommand;
using RFEventCmdPtr = std::shared_ptr<RFEvent>;

static std::mutex streamMutex;
static std::unordered_map<CUstream, RFStreamPtr> reefStreams;

static std::mutex blockingStreamMutex;
static std::unordered_map<CUstream, RFStreamPtr> blockingReefStreams;

static std::mutex eventMutex;
static std::unordered_map<CUevent, RFEventCmdPtr> reefEvents;

typedef enum CUstream_flags_enum {
    CU_STREAM_DEFAULT             = 0x0, /**< Default stream flag */
    CU_STREAM_NON_BLOCKING        = 0x1  /**< Stream does not synchronize with stream 0 (the NULL stream) */
} CUstream_flags;
HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx);

namespace reef
{

RFresult enablePreemption(CUstream cuStream, RFconfig config)
{
    RDEBUG("enable preemption on cuStream %p", cuStream);

    if (cuStream == nullptr) return REEF_ERROR_INVALID_VALUE;

    unsigned int cuStreamFlags;
    ASSERT_GPU_ERROR(cuStreamGetFlags(cuStream, &cuStreamFlags));

    std::unique_lock<std::mutex> lock(streamMutex);

    if (reefStreams.find(cuStream) != reefStreams.end()) return REEF_ERROR_ALREADY_EXIST;

    RFStreamPtr rfStream = std::make_shared<RFStream>(cuStream, config);
    reefStreams[cuStream] = rfStream;
    
    if (cuStreamFlags == CU_STREAM_DEFAULT) {
        blockingStreamMutex.lock();
        blockingReefStreams[cuStream] = rfStream;
        blockingStreamMutex.unlock();
    }

    return REEF_SUCCESS;
}

RFresult disablePreemption(CUstream cuStream)
{
    RDEBUG("disable preemption on cuStream %p", cuStream);

    if (cuStream == nullptr) return REEF_ERROR_INVALID_VALUE;

    std::unique_lock<std::mutex> lock(streamMutex);

    std::unordered_map<CUstream, RFStreamPtr>::iterator it = reefStreams.find(cuStream);
    if (it == reefStreams.end()) return REEF_ERROR_NOT_FOUND;

    RFStreamPtr rfStream = it->second;
    reefStreams.erase(it);

    blockingStreamMutex.lock();
    blockingReefStreams.erase(cuStream);
    blockingStreamMutex.unlock();

    lock.unlock();

    return rfStream->disable();
}
    
RFresult preempt(CUstream cuStream, bool synchronize)
{
    RDEBUG("preempt cuStream %p", cuStream);

    if (cuStream == nullptr) return REEF_ERROR_INVALID_VALUE;

    std::unique_lock<std::mutex> lock(streamMutex);

    std::unordered_map<CUstream, RFStreamPtr>::iterator it = reefStreams.find(cuStream);
    if (it == reefStreams.end()) return REEF_ERROR_NOT_FOUND;
    
    RFStreamPtr rfStream = it->second;
    lock.unlock();
    return rfStream->preempt(synchronize);
}

RFresult restore(CUstream cuStream)
{
    RDEBUG("restore cuStream %p", cuStream);

    if (cuStream == nullptr) return REEF_ERROR_INVALID_VALUE;

    std::unique_lock<std::mutex> lock(streamMutex);

    std::unordered_map<CUstream, RFStreamPtr>::iterator it = reefStreams.find(cuStream);
    if (it == reefStreams.end()) return REEF_ERROR_NOT_FOUND;
    
    RFStreamPtr rfStream = it->second;
    lock.unlock();
    return rfStream->restore();
}

} // namespace reef

static void syncAllRFStreams()
{
    std::list<std::shared_ptr<reef::SynchronizeCommand>> syncCmds;
    
    streamMutex.lock();
    for (auto it : reefStreams) {
        std::shared_ptr<reef::SynchronizeCommand> syncCmd = it.second->enqueueSynchronizeCommand();
        syncCmds.emplace_back(syncCmd);
    }
    streamMutex.unlock();

    for (auto syncCmd : syncCmds) {
        syncCmd->sync();
    }
}

static void syncAllBlockingStreams()
{
    std::list<std::shared_ptr<reef::SynchronizeCommand>> syncCmds;
    
    blockingStreamMutex.lock();
    for (auto it : blockingReefStreams) {
        std::shared_ptr<reef::SynchronizeCommand> syncCmd = it.second->enqueueSynchronizeCommand();
        syncCmds.emplace_back(syncCmd);
    }
    blockingStreamMutex.unlock();

    for (auto syncCmd : syncCmds) {
        syncCmd->sync();
    }
}

CUresult reefLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
{
    RDEBUG("LaunchKernel %p on cuStream %p", f, hStream);

    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (f == nullptr) goto fallback;
    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto set_arg_fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->launchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, kernelParams, extra);

unlock_fallback:
    streamMutex.unlock();

set_arg_fallback:
    return PreemptContext::setDefaultArgsThenLaunch(f, hStream, [&]()->CUresult {
        return realLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra);
    });

fallback:
    return realLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, hStream, kernelParams, extra);
}

CUresult reefMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    RDEBUG("MemcpyHtoD_v2 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

CUresult reefMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    RDEBUG("MemcpyDtoH_v2 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

CUresult reefMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    RDEBUG("MemcpyDtoD_v2 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult reefMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    RDEBUG("Memcpy2D_v2 on cuStream %p", hStream);

    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();

    return rfStream->memcpy2DAsync_v2(pCopy);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemcpy2DAsync_v2(pCopy, hStream);
}

CUresult reefMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
    RDEBUG("Memcpy3D_v2 on cuStream %p", hStream);

    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();

    return rfStream->memcpy3DAsync_v2(pCopy);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemcpy3DAsync_v2(pCopy, hStream);
}

CUresult reefMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    RDEBUG("MemsetD8 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD8Async(dstDevice, uc, N);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD8Async(dstDevice, uc, N, hStream);
}

CUresult reefMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    RDEBUG("MemsetD16 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD16Async(dstDevice, us, N);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD16Async(dstDevice, us, N, hStream);
}

CUresult reefMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    RDEBUG("MemsetD32 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD32Async(dstDevice, ui, N);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD32Async(dstDevice, ui, N, hStream);
}

CUresult reefMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("MemsetD2D8 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD2D8Async(dstDevice, dstPitch, uc, Width, Height);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult reefMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("MemsetD2D16 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD2D16Async(dstDevice, dstPitch, us, Width, Height);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult reefMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("MemsetD2D32 on cuStream %p", hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memsetD2D32Async(dstDevice, dstPitch, ui, Width, Height);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult reefMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    RDEBUG("MemFree %p on cuStream %p", (void *)dptr, hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memFreeAsync(dptr);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemFreeAsync(dptr, hStream);
}

CUresult reefMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    RDEBUG("MemAlloc %lu bytes on cuStream %p", bytesize, hStream);
    
    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();
    
    return rfStream->memAllocAsync(dptr, bytesize);

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realMemAllocAsync(dptr, bytesize, hStream);
}

CUresult reefEventRecord(CUevent hEvent, CUstream hStream)
{
    RDEBUG("EventRecord %p on cuStream %p", hEvent, hStream);

    CUresult result;
    RFStreamPtr rfStream;
    RFEventCmdPtr rfEvent;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }
    if (hEvent == nullptr) goto fallback;

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();

    rfEvent = std::make_shared<RFEvent>(hEvent, rfStream, false);
    result = rfStream->eventRecord(rfEvent);

    eventMutex.lock();
    reefEvents[hEvent] = rfEvent;
    eventMutex.unlock();

    return result;

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realEventRecord(hEvent, hStream);
}

CUresult reefEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    RDEBUG("EventRecord %p on cuStream %p", hEvent, hStream);

    CUresult result;
    RFStreamPtr rfStream;
    RFEventCmdPtr rfEvent;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }
    if (hEvent == nullptr) goto fallback;

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();

    rfEvent = std::make_shared<RFEvent>(hEvent, rfStream, false, true, flags);
    result = rfStream->eventRecord(rfEvent);

    eventMutex.lock();
    reefEvents[hEvent] = rfEvent;
    eventMutex.unlock();

    return result;

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realEventRecordWithFlags(hEvent, hStream, flags);
}

CUresult reefEventQuery(CUevent hEvent)
{
    RDEBUG("EventQuery %p", hEvent);

    RFEventCmdPtr rfEvent;
    reef::StreamCommand::CommandState state;
    std::unordered_map<CUevent, RFEventCmdPtr>::iterator it;

    if (hEvent == nullptr) goto fallback;

    eventMutex.lock();
    it = reefEvents.find(hEvent);
    if (it == reefEvents.end()) goto unlock_fallback;
    rfEvent = it->second;
    eventMutex.unlock();

    state = rfEvent->getState();
    if (state == reef::StreamCommand::Synchronized) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;

unlock_fallback:
    eventMutex.unlock();
fallback:
    return realEventQuery(hEvent);
}

CUresult reefEventSynchronize(CUevent hEvent)
{
    RDEBUG("EventSynchronize %p", hEvent);

    RFEventCmdPtr rfEvent;
    std::unordered_map<CUevent, RFEventCmdPtr>::iterator it;

    if (hEvent == nullptr) goto fallback;

    eventMutex.lock();
    it = reefEvents.find(hEvent);
    if (it == reefEvents.end()) goto unlock_fallback;
    rfEvent = it->second;
    eventMutex.unlock();

    return rfEvent->rfStream_->eventSynchronize(rfEvent);

unlock_fallback:
    eventMutex.unlock();
fallback:
    return realEventSynchronize(hEvent);
}

CUresult reefStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    RDEBUG("StreamWaitEvent %p on cuStream %p", hEvent, hStream);

    RFStreamPtr rfStream = nullptr;
    RFEventCmdPtr rfEvent = nullptr;
    std::shared_ptr<reef::EventWaitCommand> eventWaitCmd;
    std::unordered_map<CUstream, RFStreamPtr>::iterator rfStreamIt;
    std::unordered_map<CUevent, RFEventCmdPtr>::iterator rfEventIt;

    if (hEvent == nullptr) goto fallback;

    eventMutex.lock();
    rfEventIt = reefEvents.find(hEvent);
    if (rfEventIt != reefEvents.end()) rfEvent = rfEventIt->second;
    eventMutex.unlock();

    streamMutex.lock();
    rfStreamIt = reefStreams.find(hStream);
    if (rfStreamIt != reefStreams.end()) rfStream = rfStreamIt->second;
    streamMutex.unlock();

    if (rfEvent == nullptr) {
        eventWaitCmd = std::make_shared<reef::EventWaitCommand>(hEvent, Flags);
    } else {
        eventWaitCmd = std::make_shared<reef::EventWaitCommand>(rfEvent, Flags);
    }

    if (rfStream == nullptr) {
        if (hStream == nullptr) syncAllBlockingStreams();
        return eventWaitCmd->enqueue(hStream);
    } else {
        return rfStream->eventWait(eventWaitCmd);
    }

fallback:
    return realStreamWaitEvent(hStream, hEvent, Flags);
}

CUresult reefEventDestroy(CUevent hEvent)
{
    if (hEvent == nullptr) goto fallback;

    eventMutex.lock();
    reefEvents.erase(hEvent);
    eventMutex.unlock();

fallback:
    return realEventDestroy(hEvent);
}

CUresult reefEventDestroy_v2(CUevent hEvent)
{
    if (hEvent == nullptr) goto fallback;

    eventMutex.lock();
    reefEvents.erase(hEvent);
    eventMutex.unlock();

fallback:
    return realEventDestroy_v2(hEvent);
}

CUresult reefStreamSynchronize(CUstream hStream)
{
    RDEBUG("StreamSynchronize on cuStream %p", hStream);

    RFStreamPtr rfStream;
    std::unordered_map<CUstream, RFStreamPtr>::iterator it;

    if (hStream == nullptr) {
        syncAllBlockingStreams();
        goto fallback;
    }

    streamMutex.lock();
    it = reefStreams.find(hStream);
    if (it == reefStreams.end()) goto unlock_fallback;
    rfStream = it->second;
    streamMutex.unlock();

    return rfStream->synchronize();

unlock_fallback:
    streamMutex.unlock();
fallback:
    return realStreamSynchronize(hStream);
}

CUresult reefCtxSynchronize()
{
    RDEBUG("CtxSynchronize");
    syncAllRFStreams();
    return realCtxSynchronize();
}

CUresult reefInit(unsigned int Flags)
{
    static std::mutex mtx;
    static bool inited = false;

    CUresult result = realInit(Flags);

    std::unique_lock<std::mutex> lock(mtx);
    if (result == CUDA_SUCCESS && !inited) {
        CUcontext ctx = nullptr;
        ASSERT_GPU_ERROR(cuCtxGetCurrent(&ctx));

        inited = true;
        onContextInit(ctx);
    }
    
    return result;
}

CUresult reefStreamCreate(CUstream *phStream, unsigned int Flags)
{
    CUresult result = realStreamCreate(phStream, Flags);
    onStreamCreate(*phStream);
    return result;
}

CUresult reefStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority)
{
    CUresult result = realStreamCreateWithPriority(phStream, flags, priority);
    onStreamCreate(*phStream);
    return result;
}

CUresult reefStreamDestroy(CUstream hStream)
{
    CUcontext cuContext;
    if (cuCtxGetCurrent(&cuContext) == CUDA_SUCCESS && !PreemptContext::isBuiltinStream(cuContext, hStream))
        onStreamDestroy(hStream);
    return realStreamDestroy(hStream);
}

CUresult reefStreamDestroy_v2(CUstream hStream)
{
    CUcontext cuContext;
    if (cuCtxGetCurrent(&cuContext) == CUDA_SUCCESS && !PreemptContext::isBuiltinStream(cuContext, hStream))
        onStreamDestroy(hStream);
    return realStreamDestroy_v2(hStream);
}
