#include <cassert>
#include <cstring>

#include "reef/utils.h"
#include "reef/stream.h"
#include "reef/command.h"
#include "reef/reef_hook.h"
#include "reef/device_queue.h"

#define PARAM_CNT_MAX 32
#define PARAM_SIZE_MAX 3072

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);
typedef enum CUevent_flags_enum
{
    CU_EVENT_DEFAULT        = 0x0, /**< Default event flag */
    CU_EVENT_BLOCKING_SYNC  = 0x1, /**< Event uses blocking synchronization */
    CU_EVENT_DISABLE_TIMING = 0x2, /**< Event will not record timing data */
    CU_EVENT_INTERPROCESS   = 0x4  /**< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set */
} CUevent_flags;

namespace reef
{

// taskRunning do not need to be protected by TaskControlCommand->mtx_,
// because calls to taskRunning & notifyTaskArrive are protected by host_queue->mtx_,
// and taskRunning & notifyTaskComplete are called in the same backend thread
bool TaskState::taskRunning()
{
    return running_;
}

void TaskState::notifyTaskArrive(int64_t cmdIdx)
{
    std::unique_lock<std::mutex> lock(mtx_);
    lastCmdIdx_ = cmdIdx;
    if (running_) return;
    running_ = true;
    onNewStreamCommand(cuStream_);
}

void TaskState::notifyTaskComplete(int64_t cmdIdx)
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (!running_ || cmdIdx != lastCmdIdx_) return;
    // the task should be running and there is no new cmd
    running_ = false;
    onStreamSynchronized(cuStream_);
}

funcGetParamCount KernelLaunchCommand::funcGetCount_ = nullptr;
funcGetParamInfo KernelLaunchCommand::funcGetInfo_ = nullptr;

void KernelLaunchCommand::init()
{
    if (funcGetCount_ != nullptr && funcGetInfo_ != nullptr) return;

    const void *ppExportTable;
    ASSERT_GPU_ERROR(cuGetExportTable(&ppExportTable, &CU_ETID_ToolsModule));
    funcGetCount_ = *(const funcGetParamCount *)((const char *)ppExportTable + funcGetParamCountOffset);
    funcGetInfo_ = *(const funcGetParamInfo *)((const char *)ppExportTable + funcGetParamInfoOffset);
}

KernelLaunchCommand::KernelLaunchCommand(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, void **kernelParams, void **extra)
    : StreamCommand(KernelLaunch), f_(f),
      gridDimX_(gridDimX), gridDimY_(gridDimY), gridDimZ_(gridDimZ),
      blockDimX_(blockDimX), blockDimY_(blockDimY), blockDimZ_(blockDimZ),
      sharedMemBytes_(sharedMemBytes), extra_(extra)
{
    assert(funcGetCount_);
    assert(funcGetInfo_);

    size_t paramCnt;
    ASSERT_GPU_ERROR(funcGetCount_(f, &paramCnt));
    paramCnt_ = paramCnt;
    if (paramCnt_ == 0) return;

    kernelParams_ = (void **)malloc(paramCnt_ * sizeof(void *));
    // Allocate a continuous buffer for all of the params
    // buffer size = last param offset + last param size
    KernelParamInfo lastParamInfo;
    lastParamInfo.structSize_ = sizeof(lastParamInfo);
    ASSERT_GPU_ERROR(funcGetInfo_(f, paramCnt_ - 1, &lastParamInfo)); 
    size_t buffer_size = lastParamInfo.offset_ + lastParamInfo.size_;
    paramData_ = (char *)malloc(buffer_size);
    // RDEBUG("kernel param cnt: %u", paramCnt_);
    for (uint32_t i = 0; i < paramCnt_; ++ i) {
        KernelParamInfo paramInfo;
        paramInfo.structSize_ = sizeof(paramInfo); // for version checking
        ASSERT_GPU_ERROR(funcGetInfo_(f, i, &paramInfo));
        kernelParams_[i] = (void*)&paramData_[paramInfo.offset_];
        // RDEBUG("kernel param %u, size: %u", i, paramInfo.size);
        memcpy(kernelParams_[i], kernelParams[i], paramInfo.size_);
    }
}

KernelLaunchCommand::~KernelLaunchCommand()
{
    free(paramData_);
    free(kernelParams_);
}

CUresult KernelLaunchCommand::enqueue(CUstream stream)
{
    return realLaunchKernel(f_,
                            gridDimX_, gridDimY_, gridDimZ_,
                            blockDimX_, blockDimY_, blockDimZ_,
                            sharedMemBytes_, stream, kernelParams_, extra_);
}

uint64_t KernelLaunchCommand::getBlockCnt() const
{
    return (uint64_t)gridDimX_ * (uint64_t)gridDimY_ * (uint64_t)gridDimZ_;
}

EventRecordCommand::EventRecordCommand(CUevent event,
    std::shared_ptr<REEFStream> rfStream, bool builtIn, bool withFlags, unsigned int flags)
    : StreamCommand(EventRecord), event_(event), rfStream_(rfStream),
      builtIn_(builtIn), withFlags_(withFlags), flags_(flags)
{
    if (builtIn_) {
        ASSERT_GPU_ERROR(cuEventCreate(&event_, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    } else {
        mtx_ = std::make_shared<std::mutex>();
        cv_ = std::make_shared<std::condition_variable>();
    }
}

EventRecordCommand::~EventRecordCommand()
{
    if (builtIn_) {
        ASSERT_GPU_ERROR(realEventDestroy(event_));
    }
}

CUresult EventRecordCommand::enqueue(CUstream stream)
{
    if (withFlags_) return realEventRecordWithFlags(event_, stream, flags_);
    else return realEventRecord(event_, stream);
}

void EventRecordCommand::waitSubmit()
{
    if (builtIn_) assert(0);

    std::unique_lock<std::mutex> lock(*mtx_);
    while (eventState_ == Free) cv_->wait(lock);
}

StreamCommand::CommandState EventRecordCommand::getState()
{
    if (builtIn_) return eventState_;
    std::unique_lock<std::mutex> lock(*mtx_);
    return eventState_;
}

void EventRecordCommand::setState(StreamCommand::CommandState eventState)
{
    if (builtIn_) {
        eventState_ = eventState;
        return;
    }

    mtx_->lock();
    eventState_ = eventState;
    mtx_->unlock();

    cv_->notify_all();
}

CUresult EventWaitCommand::enqueue(CUstream stream)
{
    return realStreamWaitEvent(stream, event_, flags_);
}

void EventWaitCommand::waitEventSynchornize()
{
    if (isReefEvent_) rfEvent_->rfStream_->eventSynchronize(rfEvent_);
}

CUresult SynchronizeCommand::enqueue(CUstream)
{
    mtx_.lock();
    syncState_ = Synchronized;
    mtx_.unlock();

    cv_.notify_all();
    taskState_->notifyTaskComplete(cmdIdx_);
    return CUDA_SUCCESS;
}

CUresult SynchronizeCommand::sync()
{
    std::unique_lock<std::mutex> lock(mtx_);
    while (syncState_ != Synchronized) cv_.wait(lock);
    return CUDA_SUCCESS;
}

CUresult TaskControlCommand::enqueue(CUstream)
{
    taskState_->notifyTaskComplete(cmdIdx_);
    return CUDA_SUCCESS;
}

} // namespace reef