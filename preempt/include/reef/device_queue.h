#ifndef __REEF_DEVICE_QUEUE__
#define __REEF_DEVICE_QUEUE__

#include <list>
#include <queue>
#include <deque>
#include <mutex>
#include <memory>
#include <thread>
#include <condition_variable>

#include "hook/cuda_subset.h"
#include "instrument/instrument.h"
#include "reef/reef.h"

namespace reef
{

class vHostQueue;
class KernelLaunchCommand;
class EventRecordCommand;

class vDeviceQueue
{
private:
    const int64_t queueSize_;
    const int64_t batchSize_;
    const CUstream cuStream_;
    const PreemptLevel preemptLevel_;

    int64_t lastRecordedKernelIdx_ = -1;

    std::queue<std::shared_ptr<EventRecordCommand>> eventPool_;
    std::list<std::shared_ptr<EventRecordCommand>> submittedEvents_;
    std::deque<std::shared_ptr<KernelLaunchCommand>> submittedKernels_;

    std::shared_ptr<vHostQueue> vHQ_;
    std::shared_ptr<PreemptContext> preemptContext_;
    std::shared_ptr<DeviceBuffer> preemptBuffer_;

    enum WorkerState
    {
        RUNNING = 0,
        PAUSED  = 1,
        KILLED  = 2,
    };

    std::mutex mtx_;
    std::condition_variable cv_;
    WorkerState workerState_;
    std::shared_ptr<std::thread> worker_ = nullptr;
    int64_t pauseCnt_ = 0;

public:
    vDeviceQueue(int64_t queueSize, int64_t batchSize, CUstream cuStream, PreemptLevel preemptLevel,
        std::shared_ptr<vHostQueue> vHQ, std::shared_ptr<PreemptContext> preemptContext, std::shared_ptr<DeviceBuffer> preemptBuffer);
    ~vDeviceQueue();
    
    void disable();
    void pauseWorker();
    void resumeWorker(int64_t preemptKernelIdx);

    CUresult syncStream();
    CUresult syncEvent(std::shared_ptr<EventRecordCommand> eventToWait);

private:
    void loop();

    WorkerState submitKernel(std::shared_ptr<KernelLaunchCommand> kernel);
    WorkerState submitEvent(std::shared_ptr<EventRecordCommand> event);

    // just enqueue a command
    WorkerState submitNormalCommand(std::shared_ptr<StreamCommand> normalCommand);

    // when enqueue an exclusive command,
    // no other commands should be remained in device queue
    WorkerState submitExclusiveCommand(std::shared_ptr<StreamCommand> exclusiveCommand);

    // when enqueue an kernel-exclusive command,
    // no other kernels should be remained in device queue, does not require to sync the whole stream
    WorkerState submitKernelExclusiveCommand(std::shared_ptr<StreamCommand> kernelExclusiveCommand);

    std::unique_lock<std::mutex> syncStream(CUresult &result, std::unique_lock<std::mutex> lock);
    std::unique_lock<std::mutex> syncEvent(std::shared_ptr<EventRecordCommand> eventToWait, CUresult &result, std::unique_lock<std::mutex> lock);
};

}

#endif