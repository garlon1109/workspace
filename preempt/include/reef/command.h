#ifndef __REEF_COMMAND_H__
#define __REEF_COMMAND_H__

#include <mutex>
#include <memory>
#include <condition_variable>

#include "client.h"
#include "reef/reef_hook.h"
#include "hook/cufunc.h"
#include "hook/cutable.h"
#include "hook/cuda_subset.h"

namespace reef
{

class REEFStream;
class vDeviceQueue;

class TaskState
{
private:
    CUstream cuStream_;

    std::mutex mtx_;
    int64_t lastCmdIdx_ = -1;
    bool running_ = false;

public:
    TaskState(CUstream cuStream): cuStream_(cuStream) {}
    ~TaskState() {}

    bool taskRunning();
    void notifyTaskArrive(int64_t cmdIdx);
    void notifyTaskComplete(int64_t cmdIdx);
};

// This is the base class of stream commands.
class StreamCommand 
{
public:
    enum CommandState
    {
        Free            = 0,
        Submitted       = 1,
        Synchronized    = 2,
    };

    enum StreamCommandType
    {
        KernelLaunch    = 0,
        Memory          = 1,
        EventRecord     = 2,
        EventSync       = 3,
        EventWait       = 4,
        Sync            = 5,
        TaskControl     = 6,
        Disable         = 7,
    };
    const StreamCommandType cmdType_;

    StreamCommand(StreamCommandType cmdType): cmdType_(cmdType) {}
    virtual ~StreamCommand() {}
    virtual CUresult enqueue(CUstream) = 0;
};

class KernelLaunchCommand: public StreamCommand
{
public:
    CUfunction f_;
    // kernelIdx_ will start from 1 in each REEFStream
    int64_t kernelIdx_ = -1;
    bool idempotent_ = false;

private:
    uint32_t paramCnt_ = 0;
    unsigned int gridDimX_, gridDimY_, gridDimZ_;
    unsigned int blockDimX_, blockDimY_, blockDimZ_;
    unsigned int sharedMemBytes_;
    void **kernelParams_ = nullptr;
    void **extra_ = nullptr;
    char *paramData_ = nullptr;

    static funcGetParamCount funcGetCount_;
    static funcGetParamInfo funcGetInfo_;

public: 
    static void init();

    KernelLaunchCommand(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, void **kernelParams, void **extra);
    virtual ~KernelLaunchCommand();
    virtual CUresult enqueue(CUstream stream) override;
    uint64_t getBlockCnt() const;
};

class MemoryCommand: public StreamCommand
{
public:
    MemoryCommand(): StreamCommand(Memory) {}
    virtual ~MemoryCommand() {}
};

class MemcpyHtoDV2Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const void *srcHost_;
    const size_t byteCount_;

public: 
    MemcpyHtoDV2Command(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount)
        : dstDevice_(dstDevice), srcHost_(srcHost), byteCount_(byteCount) {}
    virtual ~MemcpyHtoDV2Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemcpyHtoDAsync_v2(dstDevice_, srcHost_, byteCount_, stream); }
};

class MemcpyDtoHV2Command: public MemoryCommand
{
private:
    void *dstHost_;
    const CUdeviceptr srcDevice_;
    const size_t byteCount_;

public: 
    MemcpyDtoHV2Command(void *dstHost, CUdeviceptr srcDevice, size_t byteCount)
        : dstHost_(dstHost), srcDevice_(srcDevice), byteCount_(byteCount) {}
    virtual ~MemcpyDtoHV2Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemcpyDtoHAsync_v2(dstHost_, srcDevice_, byteCount_, stream); }
};

class MemcpyDtoDV2Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const CUdeviceptr srcDevice_;
    const size_t byteCount_;

public: 
    MemcpyDtoDV2Command(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount)
        : dstDevice_(dstDevice), srcDevice_(srcDevice), byteCount_(byteCount) {}
    virtual ~MemcpyDtoDV2Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemcpyDtoDAsync_v2(dstDevice_, srcDevice_, byteCount_, stream); }
};

class Memcpy2DV2Command: public MemoryCommand
{
private:
    const CUDA_MEMCPY2D pCopy_;

public:
    Memcpy2DV2Command(const CUDA_MEMCPY2D *pCopy): pCopy_(*pCopy) {}
    virtual ~Memcpy2DV2Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemcpy2DAsync_v2(&pCopy_, stream); }
};

class Memcpy3DV2Command: public MemoryCommand
{
private:
    const CUDA_MEMCPY3D pCopy_;

public:
    Memcpy3DV2Command(const CUDA_MEMCPY3D *pCopy): pCopy_(*pCopy) {}
    virtual ~Memcpy3DV2Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemcpy3DAsync_v2(&pCopy_, stream); }
};

class MemsetD8Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const unsigned char uc_;
    const size_t N_;

public:
    MemsetD8Command(CUdeviceptr dstDevice, unsigned char uc, size_t N)
        : dstDevice_(dstDevice), uc_(uc), N_(N) {}
    virtual ~MemsetD8Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD8Async(dstDevice_, uc_, N_, stream); }
};

class MemsetD16Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const unsigned char us_;
    const size_t N_;

public:
    MemsetD16Command(CUdeviceptr dstDevice, unsigned short us, size_t N)
        : dstDevice_(dstDevice), us_(us), N_(N) {}
    virtual ~MemsetD16Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD16Async(dstDevice_, us_, N_, stream); }
};

class MemsetD32Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const unsigned char ui_;
    const size_t N_;

public:
    MemsetD32Command(CUdeviceptr dstDevice, unsigned int ui, size_t N)
        : dstDevice_(dstDevice), ui_(ui), N_(N) {}
    virtual ~MemsetD32Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD32Async(dstDevice_, ui_, N_, stream); }
};

class MemsetD2D8Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const size_t dstPitch_;
    const unsigned char uc_;
    const size_t Width_;
    const size_t Height_;

public:
    MemsetD2D8Command(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
        : dstDevice_(dstDevice), dstPitch_(dstPitch), uc_(uc), Width_(Width), Height_(Height) {}
    virtual ~MemsetD2D8Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD2D8Async(dstDevice_, dstPitch_, uc_, Width_, Height_, stream); }
};

class MemsetD2D16Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const size_t dstPitch_;
    const unsigned char us_;
    const size_t Width_;
    const size_t Height_;

public:
    MemsetD2D16Command(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
        : dstDevice_(dstDevice), dstPitch_(dstPitch), us_(us), Width_(Width), Height_(Height) {}
    virtual ~MemsetD2D16Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD2D16Async(dstDevice_, dstPitch_, us_, Width_, Height_, stream); }
};

class MemsetD2D32Command: public MemoryCommand
{
private:
    const CUdeviceptr dstDevice_;
    const size_t dstPitch_;
    const unsigned char ui_;
    const size_t Width_;
    const size_t Height_;

public:
    MemsetD2D32Command(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
        : dstDevice_(dstDevice), dstPitch_(dstPitch), ui_(ui), Width_(Width), Height_(Height) {}
    virtual ~MemsetD2D32Command() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemsetD2D32Async(dstDevice_, dstPitch_, ui_, Width_, Height_, stream); }
};

class MemFreeCommand: public MemoryCommand
{
private:
    const CUdeviceptr dptr_;

public:
    MemFreeCommand(CUdeviceptr dptr): dptr_(dptr) {}
    virtual ~MemFreeCommand() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemFreeAsync(dptr_, stream); }
};

class MemAllocCommand: public MemoryCommand
{
private:
    CUdeviceptr *dptr_;
    const size_t bytesize_;

public:
    MemAllocCommand(CUdeviceptr *dptr, size_t bytesize): dptr_(dptr), bytesize_(bytesize) {}
    virtual ~MemAllocCommand() {}
    virtual CUresult enqueue(CUstream stream) override { return realMemAllocAsync(dptr_, bytesize_, stream); }
};

class EventRecordCommand: public StreamCommand
{
public:
    CUevent event_;
    int64_t prevKernelIdx_ = -1;
    std::shared_ptr<REEFStream> rfStream_ = nullptr;
    const bool builtIn_;

private:
    const bool withFlags_;
    const unsigned int flags_;
    CommandState eventState_ = Free;

    std::shared_ptr<std::mutex> mtx_ = nullptr;
    std::shared_ptr<std::condition_variable> cv_ = nullptr;

public:
    EventRecordCommand(CUevent event, std::shared_ptr<REEFStream> rfStream,
                       bool builtIn, bool withFlags=false, unsigned int flags=0);
    virtual ~EventRecordCommand();
    virtual CUresult enqueue(CUstream stream) override;
    CUresult realSync() { return realEventSynchronize(event_); }

    void waitSubmit();
    CommandState getState();
    void setState(CommandState eventState);
};

class EventWaitCommand: public StreamCommand
{
private:
    CUevent event_ = nullptr;
    std::shared_ptr<EventRecordCommand> rfEvent_ = nullptr;
    const unsigned int flags_;
    const bool isReefEvent_;

public:
    EventWaitCommand(CUevent event, unsigned int flags)
        : StreamCommand(EventWait), event_(event), flags_(flags), isReefEvent_(false) {}
    EventWaitCommand(std::shared_ptr<EventRecordCommand> rfEvent, unsigned int flags)
        : StreamCommand(EventWait), event_(rfEvent->event_), rfEvent_(rfEvent), flags_(flags), isReefEvent_(true) {}
    virtual ~EventWaitCommand() {}
    virtual CUresult enqueue(CUstream stream) override;
    
    void waitEventSynchornize();
};

class SynchronizeCommand: public StreamCommand
{
private:
    int64_t cmdIdx_;
    std::mutex mtx_;
    std::condition_variable cv_;
    CommandState syncState_ = Free;
    std::shared_ptr<TaskState> taskState_;

public:
    SynchronizeCommand(std::shared_ptr<TaskState> taskState)
        : StreamCommand(Sync), taskState_(taskState) {}
    ~SynchronizeCommand() {}
    virtual CUresult enqueue(CUstream stream) override;

    CUresult sync();
    void setCmdIdx(int64_t cmdIdx) { cmdIdx_ = cmdIdx; }
};

class TaskControlCommand: public StreamCommand
{
private:
    int64_t cmdIdx_;
    std::shared_ptr<TaskState> taskState_;

public:
    TaskControlCommand(int64_t cmdIdx, std::shared_ptr<TaskState> taskState)
        : StreamCommand(TaskControl), cmdIdx_(cmdIdx), taskState_(taskState) {}
    ~TaskControlCommand() {}
    virtual CUresult enqueue(CUstream stream) override;
};

class DisableCommand: public StreamCommand
{
public:
    DisableCommand(): StreamCommand(Disable) {}
    ~DisableCommand() {}
    virtual CUresult enqueue(CUstream) override { return CUDA_SUCCESS; }
};

} // namespace reef

#endif