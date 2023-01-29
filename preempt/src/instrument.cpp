#include "reef/utils.h"
#include "reef/reef_hook.h"
#include "hook/cuda_hook.h"
#include "hook/cuda_subset.h"
#include "instrument/instrument.h"

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetDevice(CUdevice *device);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamDestroy(CUstream hStream);

std::mutex PreemptContext::contextMutex_;
std::unordered_map<CUcontext, std::shared_ptr<PreemptContext>> PreemptContext::preemptContexts_;

PreemptContext::PreemptContext(CUcontext cuContext, CUstream preemptStream, CUstream instrumentStream)
    : cuContext_(cuContext), preemptStream_(preemptStream),
      instrumentStream_(instrumentStream), defaultPreemptBuffer_(std::make_shared<FixedBuffer>(instrumentStream, 16))
{
    
}

PreemptContext::~PreemptContext()
{
    ASSERT_GPU_ERROR(cuStreamDestroy(preemptStream_));
    ASSERT_GPU_ERROR(cuStreamDestroy(instrumentStream_));
}

size_t PreemptContext::getRequiredSize(size_t blockCnt)
{
    return 16 + blockCnt * 8;
}

std::shared_ptr<PreemptContext> PreemptContext::create(CUcontext cuContext)
{
    std::unique_lock<std::mutex> lock(contextMutex_);
    const auto it = preemptContexts_.find(cuContext);
    if (it != preemptContexts_.end()) return it->second;

    int leastPriority, greatestPriority;
    CUstream preemptStream, instrumentStream;

    ASSERT_GPU_ERROR(cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority));

    // create preemptStream with greatest priority
    ASSERT_GPU_ERROR(realStreamCreateWithPriority(&preemptStream, 0, greatestPriority));

    // create instrumentStream with moderate priority
    ASSERT_GPU_ERROR(realStreamCreateWithPriority(&instrumentStream, 0, greatestPriority + (leastPriority > greatestPriority)));

    auto preemptContext = std::make_shared<PreemptContext>(cuContext, preemptStream, instrumentStream);
    preemptContexts_[cuContext] = preemptContext;
    return preemptContext;
}

bool PreemptContext::isBuiltinStream(CUcontext cuContext, CUstream cuStream)
{
    std::unique_lock<std::mutex> lock(contextMutex_);
    const auto it = preemptContexts_.find(cuContext);
    if (it == preemptContexts_.end()) return false;

    auto preemptContext = it->second;
    return (cuStream == preemptContext->preemptStream_) || (cuStream == preemptContext->instrumentStream_);
}

CUresult PreemptContext::setDefaultArgsThenLaunch(CUfunction func, CUstream stream, std::function<CUresult()> launch)
{
    CUcontext cuContext;
    ASSERT_GPU_ERROR(cuCtxGetCurrent(&cuContext));

    const auto preemptContext = create(cuContext);

    std::unique_lock<std::mutex> launchLock(preemptContext->launchMutex_);
    if (stream != preemptContext->instrumentStream_ && preemptContext->enabled_) {
        set_instrument_args(cuContext, func, preemptContext->defaultPreemptBuffer_->devPtr(), 0, true);
    }
    return launch();
}

void PreemptContext::enable()
{
    std::unique_lock<std::mutex> launchLock(launchMutex_);
    enabled_ = true;
}

void PreemptContext::instrument(CUfunction func)
{
    std::unique_lock<std::mutex> instrumentLock(instrumentMutex_);

    if (!instrumentedFunctions_.insert(func).second) return;

    instrument_selective(cuContext_, func, instrumentStream_);
}
    
std::shared_ptr<DeviceBuffer> PreemptContext::allocPreemptBuffer(bool useDefault)
{
    if (useDefault) return defaultPreemptBuffer_;
    
    auto preemptBuffer = std::make_shared<ResizableBuffer>(instrumentStream_);
    return preemptBuffer;
}

CUresult PreemptContext::setArgsThenLaunch(CUfunction func, CUdeviceptr preemptPtr, int64_t kernelIdx, bool idempotent, std::function<CUresult()> launch)
{
    std::unique_lock<std::mutex> launchLock(launchMutex_);
    set_instrument_args(cuContext_, func, preemptPtr, kernelIdx, idempotent);
    return launch();
}

void PreemptContext::resetPreemptFlag(CUdeviceptr preemptPtr)
{
    ASSERT_GPU_ERROR(realMemsetD8Async(preemptPtr, 0, 16, preemptStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(preemptStream_));
}

void PreemptContext::setPreemptFlag(CUdeviceptr preemptPtr, uint8_t preemptFlag)
{
    ASSERT_GPU_ERROR(realMemsetD8Async(preemptPtr + 7, preemptFlag, 1, preemptStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(preemptStream_));
}

uint8_t PreemptContext::getPreemptFlag(CUdeviceptr preemptPtr, int64_t &preemptIdx)
{
    char buf[16];
    ASSERT_GPU_ERROR(realMemcpyDtoHAsync_v2(buf, preemptPtr, 16, preemptStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(preemptStream_));

    uint8_t preemptFlag = *(uint8_t *)(buf + 7);
    preemptIdx = (int64_t)*(uint64_t *)(buf + 8);

    return preemptFlag;
}

uint8_t PreemptContext::getThenResetPreemptFlag(CUdeviceptr preemptPtr, int64_t &preemptIdx)
{
    char buf[16];
    ASSERT_GPU_ERROR(realMemcpyDtoHAsync_v2(buf, preemptPtr, 16, preemptStream_));
    ASSERT_GPU_ERROR(realMemsetD8Async(preemptPtr, 0, 16, preemptStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(preemptStream_));

    uint8_t preemptFlag = *(uint8_t *)(buf + 7);
    preemptIdx = (int64_t)*(uint64_t *)(buf + 8);

    return preemptFlag;
}
