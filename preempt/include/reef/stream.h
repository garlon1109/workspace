#ifndef __REEF_STREAM_H__
#define __REEF_STREAM_H__

#include <mutex>
#include <memory>
#include <condition_variable>

#include "hook/cuda_subset.h"
#include "reef/reef.h"
#include "reef/command.h"
#include "reef/host_queue.h"
#include "reef/device_queue.h"
#include "instrument/instrument.h"

namespace reef
{

class REEFStream
{
private:
    const CUstream cuStream_;
    const PreemptLevel preemptLevel_;

    std::shared_ptr<PreemptContext> preemptContext_;
    std::shared_ptr<DeviceBuffer> preemptBuffer_;

    std::shared_ptr<vHostQueue> vHQ_;
    std::shared_ptr<vDeviceQueue> vDQ_;
    std::shared_ptr<TaskState> taskState_;

    std::mutex mtx_;
    int64_t lastKernelIdx_ = 0;

public:
    REEFStream(CUstream cuStream, RFconfig config);
    ~REEFStream();

    reef::RFresult preempt(bool synchronize);
    reef::RFresult restore();
    reef::RFresult disable();

    CUresult launchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, void** kernelParams, void** extra);
    
    CUresult memcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount);
    CUresult memcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount);
    CUresult memcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount);
    CUresult memcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy);
    CUresult memcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy);
    
    CUresult memsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N);
    CUresult memsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N);
    CUresult memsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N);
    CUresult memsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
    CUresult memsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
    CUresult memsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);

    CUresult memFreeAsync(CUdeviceptr dptr);
    CUresult memAllocAsync(CUdeviceptr *dptr, size_t bytesize);

    CUresult eventRecord(std::shared_ptr<EventRecordCommand> rfEvent);
    CUresult eventSynchronize(std::shared_ptr<EventRecordCommand> rfEvent);
    CUresult eventWait(std::shared_ptr<EventWaitCommand> eventWaitCmd);
    CUresult synchronize();
    std::shared_ptr<SynchronizeCommand> enqueueSynchronizeCommand();
};

} // namespace reef
#endif