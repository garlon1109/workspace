#include <chrono>
#include <numeric>

#include "reef/command.h"
#include "reef/host_queue.h"

namespace reef
{

vHostQueue::vHostQueue(int32_t taskTimeout, std::shared_ptr<TaskState> taskState)
    : enableTimeout_(taskTimeout >= 0), taskTimeout_(taskTimeout), taskState_(taskState)
{
    
}

vHostQueue::~vHostQueue()
{
    
}

void vHostQueue::enqueueAll(CUstream stream)
{
    std::unique_lock<std::mutex> lock(mtx_);
    while (commands_.size() > 0) {
        std::shared_ptr<StreamCommand> cmd = commands_.front();
        commands_.pop_front();
        cmd->enqueue(stream);
    }
}

void vHostQueue::insertFront(std::shared_ptr<StreamCommand> command)
{
    mtx_.lock();
    commands_.emplace_front(command);
    mtx_.unlock();

    cv_.notify_all();
}

CUresult vHostQueue::produceCommand(std::shared_ptr<StreamCommand> command)
{
    mtx_.lock();
    int64_t cmdIdx = ++lastCmdIdx_;
    commands_.emplace_back(command);
    taskState_->notifyTaskArrive(cmdIdx);
    mtx_.unlock();

    cv_.notify_all();
    return CUDA_SUCCESS;
}

CUresult vHostQueue::produceSynchronizeCommand(std::shared_ptr<SynchronizeCommand> syncCmd)
{
    mtx_.lock();
    int64_t cmdIdx = ++lastCmdIdx_;
    syncCmd->setCmdIdx(cmdIdx);
    commands_.emplace_back(syncCmd);
    taskState_->notifyTaskArrive(cmdIdx);
    mtx_.unlock();

    cv_.notify_all();
    return CUDA_SUCCESS;
}

std::shared_ptr<StreamCommand> vHostQueue::consumeCommand()
{
    std::cv_status timeout = std::cv_status::no_timeout;
    std::unique_lock<std::mutex> lock(mtx_);
    while (commands_.size() == 0) {
        if (!enableTimeout_ || !taskState_->taskRunning()) {
            // did not enable timeout mechanism, or the task is not running
            // wait without timeout until new command enqueue
            cv_.wait(lock);
            continue;
        }

        if (timeout != std::cv_status::timeout) {
            timeout = cv_.wait_for(lock, taskTimeout_);
            continue;
        }

        // enabled timeout mechanism, and the task is running, and has timeout
        // enqueue a TaskControlCommand into device queue
        int64_t cmdIdx = ++lastCmdIdx_;
        taskState_->notifyTaskArrive(cmdIdx);
        auto taskControlCommand = std::make_shared<TaskControlCommand>(cmdIdx, taskState_);
        return taskControlCommand;
    }

    std::shared_ptr<StreamCommand> command = commands_.front();
    commands_.pop_front();
    return command;
}

} // namespace reef
