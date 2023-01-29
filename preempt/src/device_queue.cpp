#include <cassert>

#include "client.h"
#include "hook/cuda_hook.h"
#include "hook/cuda_subset.h"
#include "reef/utils.h"
#include "reef/command.h"
#include "reef/reef_hook.h"
#include "reef/host_queue.h"
#include "reef/device_queue.h"
#include "instrument/instrument.h"

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventDestroy(CUevent phEvent);

typedef enum CUevent_flags_enum
{
    CU_EVENT_DEFAULT        = 0x0, /**< Default event flag */
    CU_EVENT_BLOCKING_SYNC  = 0x1, /**< Event uses blocking synchronization */
    CU_EVENT_DISABLE_TIMING = 0x2, /**< Event will not record timing data */
    CU_EVENT_INTERPROCESS   = 0x4  /**< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set */
} CUevent_flags;

namespace reef
{

vDeviceQueue::vDeviceQueue(int64_t queueSize, int64_t batchSize, CUstream cuStream, PreemptLevel preemptLevel,
    std::shared_ptr<vHostQueue> vHQ, std::shared_ptr<PreemptContext> preemptContext, std::shared_ptr<DeviceBuffer> preemptBuffer)
    : queueSize_(queueSize), batchSize_(batchSize), cuStream_(cuStream), preemptLevel_(preemptLevel),
      vHQ_(vHQ), preemptContext_(preemptContext), preemptBuffer_(preemptBuffer)
{
    // queue size must not be smaller than batch size
    // otherwise, there could be no event to wait if the queue is full
    assert(queueSize >= batchSize);

    // Init the cuda events pool
    for (int64_t i = 0; i < queueSize_; i++) {
        eventPool_.push(std::make_shared<EventRecordCommand>(nullptr, nullptr, true));
    }

    workerState_ = RUNNING;
    assert(worker_ == nullptr);
    worker_ = std::make_shared<std::thread>([&](){
        ASSERT_GPU_ERROR(cuCtxSetCurrent(preemptContext_->cuContext_));
        this->loop();
    });
}

vDeviceQueue::~vDeviceQueue()
{
    
}

void vDeviceQueue::disable()
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (workerState_ == KILLED) return;
    
    workerState_ = KILLED;
    lock.unlock();

    cv_.notify_all();
    worker_->join();
}
    
void vDeviceQueue::pauseWorker()
{
    std::unique_lock<std::mutex> lock(mtx_);
    workerState_ = PAUSED;
    pauseCnt_++;
}

void vDeviceQueue::resumeWorker(int64_t preemptKernelIdx)
{
    std::unique_lock<std::mutex> lock(mtx_);

    if (preemptKernelIdx < 0) {
        // nothing was killed, all kernels have been executed
        // pop and delete all kernels launched
        while (submittedKernels_.size() > 0) {
            submittedKernels_.pop_front();
        }

        while (submittedEvents_.size() > 0) {
            auto submittedEvent = submittedEvents_.front();

            // notify the thread calling cuEventSynchronize(submittedEvent)
            submittedEvent->setState(StreamCommand::Synchronized);
            
            if (submittedEvent->builtIn_) {
                // recycle built-in event
                eventPool_.push(submittedEvent);
            }

            submittedEvents_.pop_front();
        }

        workerState_ = RUNNING;
        lock.unlock();
        cv_.notify_all();
        return;
    }

    // re-launch all the killed kernels
    // re-submit all events that have not been recorded
    const int64_t queueLength = submittedKernels_.size();

    // if queueLenght == 0, then no kernel is running when preempted,
    // and no kernel is killed, preempt flag will be 1 and preemptKernelIdx will be -1
    assert(queueLength > 0);

    const int64_t frontKernelIdx = submittedKernels_.front()->kernelIdx_;
    int64_t reLaunchKernelPos = preemptKernelIdx - frontKernelIdx;
    assert(reLaunchKernelPos >= 0);
    assert(reLaunchKernelPos < queueLength);

    for (auto event : submittedEvents_) {
        if (event->prevKernelIdx_ < preemptKernelIdx)  {
            event->setState(StreamCommand::Synchronized);
            continue;
        }

        int64_t eventPos = event->prevKernelIdx_ - frontKernelIdx;
        assert(eventPos >= 0);
        assert(eventPos < queueLength);

        for ( ; reLaunchKernelPos <= eventPos; ++reLaunchKernelPos) {
            auto kernel = submittedKernels_[reLaunchKernelPos];

            RDEBUG("re-launch kernel %ld", kernel->kernelIdx_);
            ASSERT_GPU_ERROR(preemptContext_->setArgsThenLaunch(kernel->f_, preemptBuffer_->devPtr(), kernel->kernelIdx_, kernel->idempotent_, [&]()->CUresult {
                return kernel->enqueue(cuStream_);
            }));
        }

        RDEBUG("re-submit event");
        event->enqueue(cuStream_);
    }

    for ( ; reLaunchKernelPos < queueLength; ++reLaunchKernelPos) {
        auto kernel = submittedKernels_[reLaunchKernelPos];

        RDEBUG("re-launch kernel %ld", kernel->kernelIdx_);
        ASSERT_GPU_ERROR(preemptContext_->setArgsThenLaunch(kernel->f_, preemptBuffer_->devPtr(), kernel->kernelIdx_, kernel->idempotent_, [&]()->CUresult {
            return kernel->enqueue(cuStream_);
        }));
    }

    workerState_ = RUNNING;
    lock.unlock();
    cv_.notify_all();
}

CUresult vDeviceQueue::syncStream()
{
    CUresult result = CUDA_SUCCESS;
    std::unique_lock<std::mutex> lock(mtx_);
    this->syncStream(result, std::move(lock));
    return result;
}

CUresult vDeviceQueue::syncEvent(std::shared_ptr<EventRecordCommand> eventToWait)
{
    CUresult result = CUDA_SUCCESS;
    std::unique_lock<std::mutex> lock(mtx_);
    this->syncEvent(eventToWait, result, std::move(lock));
    return result;
}

void vDeviceQueue::loop()
{
    while (true) {
        std::shared_ptr<StreamCommand> cmd = vHQ_->consumeCommand();

        switch (cmd->cmdType_)
        {
        case StreamCommand::KernelLaunch:
            if (submitKernel(std::static_pointer_cast<KernelLaunchCommand>(cmd)) == KILLED) return;
            break;
        
        case StreamCommand::Memory:
            if (submitKernelExclusiveCommand(cmd) == KILLED) return;
            break;
        
        case StreamCommand::EventRecord:
            if (submitEvent(std::static_pointer_cast<EventRecordCommand>(cmd)) == KILLED) return;
            break;

        case StreamCommand::EventWait:
            std::static_pointer_cast<EventWaitCommand>(cmd)->waitEventSynchornize();
            if (submitKernelExclusiveCommand(cmd) == KILLED) return;
            break;
        
        case StreamCommand::Sync:
            if (submitExclusiveCommand(cmd) == KILLED) return;
            break;
        
        case StreamCommand::TaskControl:
            if (submitExclusiveCommand(cmd) == KILLED) return;
            break;
        
        case StreamCommand::Disable:
            return;

        default:
            break;

        }
    }
}

vDeviceQueue::WorkerState vDeviceQueue::submitKernel(std::shared_ptr<KernelLaunchCommand> kernel)
{
    std::unique_lock<std::mutex> lock(mtx_);

    if (submittedKernels_.size() >= (size_t)queueSize_) {
        // the vDQ is full, wait for a slot
        std::shared_ptr<EventRecordCommand> eventToWait = nullptr;
        int64_t frontKernelIdx = submittedKernels_.front()->kernelIdx_;
        
        for (auto event : submittedEvents_) {
            // make sure after syncEvent(eventToWait),
            // there will be at least one slot
            if (event->prevKernelIdx_ >= frontKernelIdx) {
                eventToWait = event;
                break;
            }
        }

        CUresult result = CUDA_SUCCESS;
        assert(eventToWait);
        // syncEvent() will wait if the worker is paused
        lock = syncEvent(eventToWait, result, std::move(lock));

    } else {
        // wait if the worker is paused
        while (true) {
            if (workerState_ == RUNNING) break;
            else if (workerState_ == PAUSED) cv_.wait(lock);
            else if (workerState_ == KILLED) break;
            else assert(0);
        }
    }
    if (preemptLevel_ >= PreemptDeviceQueue) {
        ASSERT_GPU_ERROR(preemptContext_->setArgsThenLaunch(kernel->f_, preemptBuffer_->devPtr(), kernel->kernelIdx_, kernel->idempotent_, [&]()->CUresult {
            return kernel->enqueue(cuStream_);
        }));
    }
    else 
        kernel->enqueue(cuStream_);
    submittedKernels_.emplace_back(kernel);

    if (kernel->kernelIdx_ - lastRecordedKernelIdx_ == batchSize_) {
        // submit a event after this kernel
        assert(eventPool_.size() > 0);
        auto eventCmd = eventPool_.front();
        eventPool_.pop();

        eventCmd->prevKernelIdx_ = kernel->kernelIdx_;
        eventCmd->enqueue(cuStream_);

        submittedEvents_.emplace_back(eventCmd);
        lastRecordedKernelIdx_ = eventCmd->prevKernelIdx_;
        eventCmd->setState(StreamCommand::Submitted);
    }

    WorkerState workerState = workerState_;
    return workerState;
}

vDeviceQueue::WorkerState vDeviceQueue::submitEvent(std::shared_ptr<EventRecordCommand> event)
{
    std::unique_lock<std::mutex> lock(mtx_);
    
    event->enqueue(cuStream_);
    
    submittedEvents_.emplace_back(event);
    lastRecordedKernelIdx_ = event->prevKernelIdx_;
    event->setState(StreamCommand::Submitted);
    
    return workerState_;
}

vDeviceQueue::WorkerState vDeviceQueue::submitNormalCommand(std::shared_ptr<StreamCommand> normalCommand)
{
    std::unique_lock<std::mutex> lock(mtx_);
    // wait if the worker is paused
    while (true) {
        if (workerState_ == RUNNING) break;
        else if (workerState_ == PAUSED) cv_.wait(lock);
        else if (workerState_ == KILLED) break;
        else assert(0);
    }
    
    normalCommand->enqueue(cuStream_);
    return workerState_;
}

vDeviceQueue::WorkerState vDeviceQueue::submitExclusiveCommand(std::shared_ptr<StreamCommand> exclusiveCommand)
{
    CUresult result = CUDA_SUCCESS;
    std::unique_lock<std::mutex> lock(mtx_);

    // syncStream() will wait if the worker is paused
    lock = syncStream(result, std::move(lock));
    
    exclusiveCommand->enqueue(cuStream_);
    return workerState_;
}

vDeviceQueue::WorkerState vDeviceQueue::submitKernelExclusiveCommand(std::shared_ptr<StreamCommand> kernelExclusiveCommand)
{
    CUresult result = CUDA_SUCCESS;
    std::unique_lock<std::mutex> lock(mtx_);

    if (submittedKernels_.size() > 0) {
        // there are kernels in device queue, wait until they finish
        // syncStream() will wait if the worker is paused
        lock = syncStream(result, std::move(lock));

    } else {
        // fast path, will not call an extra cuStreamSynchronize()
        // wait if the worker is paused
        while (true) {
            if (workerState_ == RUNNING) break;
            else if (workerState_ == PAUSED) cv_.wait(lock);
            else if (workerState_ == KILLED) break;
            else assert(0);
        }
    }

    kernelExclusiveCommand->enqueue(cuStream_);
    return workerState_;
}

std::unique_lock<std::mutex> vDeviceQueue::syncStream(CUresult &result, std::unique_lock<std::mutex> lock)
{
    while (true) {
        // wait if worker is paused
        while (true) {
            if (workerState_ == RUNNING) break;
            else if (workerState_ == PAUSED) cv_.wait(lock);
            else if (workerState_ == KILLED) break;
            else assert(0);
        }

        int64_t pauseCnt = pauseCnt_;

        lock.unlock();
        result = realStreamSynchronize(cuStream_);
        lock.lock();

        // if no preemption happened during realStreamSynchronize(), then break;
        if (pauseCnt_ == pauseCnt) break;
    }

    // pop and delete all kernels launched
    while (submittedKernels_.size() > 0) {
        submittedKernels_.pop_front();
    }

    while (submittedEvents_.size() > 0) {
        auto submittedEvent = submittedEvents_.front();

        // notify the thread calling cuEventSynchronize(submittedEvent)
        submittedEvent->setState(StreamCommand::Synchronized);
        
        if (submittedEvent->builtIn_) {
            // recycle built-in event
            eventPool_.push(submittedEvent);
        }

        submittedEvents_.pop_front();
    }

    return lock;
}

std::unique_lock<std::mutex> vDeviceQueue::syncEvent(std::shared_ptr<EventRecordCommand> eventToWait,
    CUresult &result, std::unique_lock<std::mutex> lock)
{
    while (true) {
        // wait if worker is paused
        while (true) {
            if (workerState_ == RUNNING) break;
            else if (workerState_ == PAUSED) cv_.wait(lock);
            else if (workerState_ == KILLED) break;
            else assert(0);
        }

        StreamCommand::CommandState eventState = eventToWait->getState();
        if (eventState == StreamCommand::Free) return lock;
        if (eventState == StreamCommand::Synchronized) break;

        int64_t pauseCnt = pauseCnt_;

        lock.unlock();
        result = eventToWait->realSync();
        lock.lock();

        // if no preemption happened during realEventSynchronize(), then break;
        if (pauseCnt_ == pauseCnt) break;
    }
    
    int64_t prevKernelIdx = eventToWait->prevKernelIdx_;

    // pop and delete kernels launched before eventToWait
    while (submittedKernels_.size() > 0) {
        auto completedKernel = submittedKernels_.front();
        if (completedKernel->kernelIdx_ > prevKernelIdx) break;

        submittedKernels_.pop_front();
    }

    while (submittedEvents_.size() > 0) {
        auto submittedEvent = submittedEvents_.front();
        if (submittedEvent->prevKernelIdx_ > prevKernelIdx) break;

        // notify the thread calling cuEventSynchronize(submittedEvent)
        submittedEvent->setState(StreamCommand::Synchronized);
        
        if (submittedEvent->builtIn_) {
            // recycle built-in event
            eventPool_.push(submittedEvent);
        }

        submittedEvents_.pop_front();
    }

    return lock;
}

}
