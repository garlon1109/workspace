#include <assert.h>

#include "client.h"
#include "reef/utils.h"
#include "reef/stream.h"
#include "reef/reef_hook.h"
#include "instrument/instrument.h"

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx);

namespace reef
{

REEFStream::REEFStream(CUstream cuStream, RFconfig config): cuStream_(cuStream),
    preemptLevel_(config.preemptLevel), taskState_(std::make_shared<TaskState>(cuStream))
{
    KernelLaunchCommand::init();

    // make sure no tasks are running on cuStream_
    ASSERT_GPU_ERROR(realStreamSynchronize(cuStream_));
    
    CUcontext cuContext;
    ASSERT_GPU_ERROR(cuCtxGetCurrent(&cuContext));

    preemptContext_ = PreemptContext::create(cuContext);
    preemptContext_->enable();
    preemptBuffer_ = preemptContext_->allocPreemptBuffer(preemptLevel_ < PreemptDeviceQueue);

    vHQ_ = std::make_shared<vHostQueue>(config.taskTimeout, taskState_);
    vDQ_ = std::make_shared<vDeviceQueue>(config.queueSize, config.batchSize, cuStream_, preemptLevel_, vHQ_, preemptContext_, preemptBuffer_);
}

REEFStream::~REEFStream()
{
    disable();
}

RFresult REEFStream::preempt(bool synchronize)
{
    vDQ_->pauseWorker();

    if (preemptLevel_ >= PreemptDeviceQueue) preemptContext_->setPreemptFlag(preemptBuffer_->devPtr(), 1);
    
    if (synchronize) ASSERT_GPU_ERROR(realStreamSynchronize(cuStream_));

    return REEF_SUCCESS;
}

RFresult REEFStream::restore()
{
    ASSERT_GPU_ERROR(realStreamSynchronize(cuStream_));

    int64_t preemptKernelIdx = -1;

    if (preemptLevel_ >= PreemptDeviceQueue) {
        uint8_t preemptFlag = preemptContext_->getThenResetPreemptFlag(preemptBuffer_->devPtr(), preemptKernelIdx);
        RDEBUG("restore preempt flag: %u, preempt kernel idx: %ld", preemptFlag, preemptKernelIdx);
        if (preemptFlag == 1) preemptKernelIdx = -1;
    }

    vDQ_->resumeWorker(preemptKernelIdx);

    return REEF_SUCCESS;
}

RFresult REEFStream::disable()
{
    std::shared_ptr<DisableCommand> cmd = std::make_shared<DisableCommand>();
    vHQ_->insertFront(cmd);
    vDQ_->disable();
    vHQ_->enqueueAll(cuStream_);
    return REEF_SUCCESS;
}

CUresult REEFStream::launchKernel(CUfunction f,
                                  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                  unsigned int sharedMemBytes, void** kernelParams, void** extra)
{
    if (preemptLevel_ >= PreemptDeviceQueue) preemptContext_->instrument(f);

    std::shared_ptr<KernelLaunchCommand> cmd = std::make_shared<KernelLaunchCommand>(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, kernelParams, extra);

    std::unique_lock<std::mutex> lock(mtx_);
    cmd->kernelIdx_ = ++lastKernelIdx_;

    if (preemptLevel_ >= PreemptDeviceQueue) {
        uint64_t blockCnt = cmd->getBlockCnt();
        preemptBuffer_->expand(PreemptContext::getRequiredSize(blockCnt));
    } else {
        // will use default preempt ptr when launch
        // does not need to enable block-level preemption
        cmd->idempotent_ = true;
    }

    lock.unlock();
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount)
{
    std::shared_ptr<MemcpyHtoDV2Command> cmd = std::make_shared<MemcpyHtoDV2Command>(dstDevice, srcHost, byteCount);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount)
{
    std::shared_ptr<MemcpyDtoHV2Command> cmd = std::make_shared<MemcpyDtoHV2Command>(dstHost, srcDevice, byteCount);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount)
{
    std::shared_ptr<MemcpyDtoDV2Command> cmd = std::make_shared<MemcpyDtoDV2Command>(dstDevice, srcDevice, byteCount);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy)
{
    std::shared_ptr<Memcpy2DV2Command> cmd = std::make_shared<Memcpy2DV2Command>(pCopy);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy)
{
    std::shared_ptr<Memcpy3DV2Command> cmd = std::make_shared<Memcpy3DV2Command>(pCopy);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    std::shared_ptr<MemsetD8Command> cmd = std::make_shared<MemsetD8Command>(dstDevice, uc, N);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    std::shared_ptr<MemsetD16Command> cmd = std::make_shared<MemsetD16Command>(dstDevice, us, N);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    std::shared_ptr<MemsetD32Command> cmd = std::make_shared<MemsetD32Command>(dstDevice, ui, N);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    std::shared_ptr<MemsetD2D8Command> cmd = std::make_shared<MemsetD2D8Command>(dstDevice, dstPitch, uc, Width, Height);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    std::shared_ptr<MemsetD2D16Command> cmd = std::make_shared<MemsetD2D16Command>(dstDevice, dstPitch, us, Width, Height);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    std::shared_ptr<MemsetD2D32Command> cmd = std::make_shared<MemsetD2D32Command>(dstDevice, dstPitch, ui, Width, Height);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memFreeAsync(CUdeviceptr dptr)
{
    std::shared_ptr<MemFreeCommand> cmd = std::make_shared<MemFreeCommand>(dptr);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::memAllocAsync(CUdeviceptr *dptr, size_t bytesize)
{
    std::shared_ptr<MemAllocCommand> cmd = std::make_shared<MemAllocCommand>(dptr, bytesize);
    return vHQ_->produceCommand(cmd);
}

CUresult REEFStream::eventRecord(std::shared_ptr<EventRecordCommand> rfEvent)
{
    std::unique_lock<std::mutex> lock(mtx_);
    rfEvent->prevKernelIdx_ = lastKernelIdx_;
    return vHQ_->produceCommand(rfEvent);
}

CUresult REEFStream::eventSynchronize(std::shared_ptr<EventRecordCommand> rfEvent)
{
    if (rfEvent->getState() == StreamCommand::Synchronized) return CUDA_SUCCESS;
    rfEvent->waitSubmit();

    return vDQ_->syncEvent(rfEvent);
}

CUresult REEFStream::eventWait(std::shared_ptr<EventWaitCommand> eventWaitCmd)
{
    return vHQ_->produceCommand(eventWaitCmd);
}

CUresult REEFStream::synchronize()
{
    std::shared_ptr<SynchronizeCommand> cmd = enqueueSynchronizeCommand();
    return cmd->sync();
}

std::shared_ptr<SynchronizeCommand> REEFStream::enqueueSynchronizeCommand()
{
    std::shared_ptr<SynchronizeCommand> cmd = std::make_shared<SynchronizeCommand>(taskState_);
    vHQ_->produceSynchronizeCommand(cmd);
    return cmd;
}

} // namespace reef
