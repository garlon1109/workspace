#ifndef __REEF_REEF_HOOK_H__
#define __REEF_REEF_HOOK_H__

#include "hook/cuda_subset.h"

// ############# kernel related #############
CUresult reefLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
CUresult realLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

// ############# memory related #############
CUresult reefMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult realMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);

CUresult reefMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult realMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

CUresult reefMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult realMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

CUresult reefMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
CUresult realMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);

CUresult reefMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
CUresult realMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);

CUresult reefMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult realMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);

CUresult reefMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
CUresult realMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);

CUresult reefMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
CUresult realMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);

CUresult reefMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
CUresult realMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);

CUresult reefMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
CUresult realMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);

CUresult reefMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
CUresult realMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);

CUresult reefMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult realMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

CUresult reefMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult realMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);

// ############# event related #############
CUresult reefEventRecord(CUevent hEvent, CUstream hStream);
CUresult realEventRecord(CUevent hEvent, CUstream hStream);

CUresult reefEventQuery(CUevent hEvent);
CUresult realEventQuery(CUevent hEvent);

CUresult reefEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
CUresult realEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);

CUresult reefEventSynchronize(CUevent hEvent);
CUresult realEventSynchronize(CUevent hEvent);

CUresult reefStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult realStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);

CUresult reefEventDestroy(CUevent hEvent);
CUresult realEventDestroy(CUevent hEvent);

CUresult reefEventDestroy_v2(CUevent hEvent);
CUresult realEventDestroy_v2(CUevent hEvent);

// ############# stream related #############
CUresult reefStreamSynchronize(CUstream hStream);
CUresult realStreamSynchronize(CUstream hStream);

CUresult reefCtxSynchronize();
CUresult realCtxSynchronize();

// ############# scheduler client related #############
CUresult reefInit(unsigned int Flags);
CUresult realInit(unsigned int Flags);

CUresult reefStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult realStreamCreate(CUstream *phStream, unsigned int Flags);

CUresult reefStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult realStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);

CUresult reefStreamDestroy(CUstream hStream);
CUresult realStreamDestroy(CUstream hStream);

CUresult reefStreamDestroy_v2(CUstream hStream);
CUresult realStreamDestroy_v2(CUstream hStream);

#endif
