#include "reef/utils.h"
#include "reef/reef_hook.h"
#include "hook/cuda_hook.h"

CUresult realLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
{
    RDEBUG("realLaunchKernel %p on cuStream %p", f, hStream);
    using func_ptr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **, void **);
    static auto cuLaunchKernel = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchKernel"));
    HOOK_CHECK(cuLaunchKernel);
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                          sharedMemBytes, hStream, kernelParams, extra);
}

CUresult realMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    RDEBUG("realMemcpyHtoD_v2 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoDAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcHost, ByteCount, hStream);
}

CUresult realMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    RDEBUG("realMemcpyDtoH_v2 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoHAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcDevice, ByteCount, hStream);
}

CUresult realMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    RDEBUG("realMemcpyDtoD_v2 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoDAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult realMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) 
{
    RDEBUG("realMemcpy2D_v2 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D*, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2DAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

CUresult realMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
    RDEBUG("realMemcpy3D_v2 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3DAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

CUresult realMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    RDEBUG("realMemsetD8 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD8Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, uc, N, hStream);
}

CUresult realMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    RDEBUG("realMemsetD16 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD16Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, us, N, hStream);
}

CUresult realMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    RDEBUG("realMemsetD32 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD32Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, ui, N, hStream);
}

CUresult realMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("realMemsetD2D8 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D8Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult realMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("realMemsetD2D16 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D16Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult realMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    RDEBUG("realMemsetD2D32 on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D32Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult realMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    RDEBUG("realMemFree %p on cuStream %p", (void *)dptr, hStream);
    using func_ptr = CUresult (*)(CUdeviceptr, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemFreeAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, hStream);
}

CUresult realMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    RDEBUG("realMemAlloc %lu bytes on cuStream %p", bytesize, hStream);
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize, hStream);
}

CUresult realEventRecord(CUevent hEvent, CUstream hStream)
{
    RDEBUG("realEventRecord %p on cuStream %p", hEvent, hStream);
    using func_ptr = CUresult (*)(CUevent, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventRecord"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent, hStream);
}

CUresult realEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    RDEBUG("realEventRecord %p on cuStream %p", hEvent, hStream);
    using func_ptr = CUresult (*)(CUevent, CUstream, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventRecordWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent, hStream, flags);
}

CUresult realEventQuery(CUevent hEvent)
{
    RDEBUG("realEventQuery %p", hEvent);
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventQuery"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

CUresult realEventSynchronize(CUevent hEvent)
{
    RDEBUG("realEventSynchronize %p", hEvent);
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

CUresult realStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    RDEBUG("realStreamWaitEvent %p on cuStream %p", hEvent, hStream);
    using func_ptr = CUresult (*)(CUstream, CUevent, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWaitEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, hEvent, Flags);
}

CUresult realEventDestroy(CUevent hEvent)
{
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

CUresult realEventDestroy_v2(CUevent hEvent)
{
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

CUresult realStreamSynchronize(CUstream hStream)
{
    RDEBUG("realStreamSynchronize on cuStream %p", hStream);
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

CUresult realCtxSynchronize()
{
    RDEBUG("realCtxSynchronize");
    using func_ptr = CUresult (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

CUresult realInit(unsigned int Flags)
{
    using func_ptr = CUresult (*)(unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuInit"));
    HOOK_CHECK(func_entry);
    return func_entry(Flags);
}

CUresult realStreamCreate(CUstream *phStream, unsigned int Flags)
{
    using func_ptr = CUresult (*)(CUstream *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(phStream, Flags);
}

CUresult realStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority)
{
    using func_ptr = CUresult (*)(CUstream *, unsigned int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamCreateWithPriority"));
    HOOK_CHECK(func_entry);
    return func_entry(phStream, flags, priority);
}

CUresult realStreamDestroy(CUstream hStream)
{
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

CUresult realStreamDestroy_v2(CUstream hStream)
{
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}
