#ifndef __REEF_HOST_QUEUE__
#define __REEF_HOST_QUEUE__

#include <list>
#include <memory>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include "hook/cuda_subset.h"

namespace reef
{

class TaskState;
class StreamCommand;
class SynchronizeCommand;

// vHostQueue buffers the intercepted cuda commands (e.g., kernel launch).
// The commands will be submitted to the CUDA driver by a background thread.
class vHostQueue
{
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    int64_t lastCmdIdx_ = -1;
    const bool enableTimeout_;
    const std::chrono::microseconds taskTimeout_;
    const std::shared_ptr<TaskState> taskState_;
    std::list<std::shared_ptr<StreamCommand>> commands_;

public:
    vHostQueue(int32_t taskTimeout, std::shared_ptr<TaskState> taskState);
    ~vHostQueue();
    
    void enqueueAll(CUstream stream);
    void insertFront(std::shared_ptr<StreamCommand> command);

    CUresult produceCommand(std::shared_ptr<StreamCommand> command);
    CUresult produceSynchronizeCommand(std::shared_ptr<SynchronizeCommand> syncCmd);
    std::shared_ptr<StreamCommand> consumeCommand();
};

} // namespace reef

#endif

