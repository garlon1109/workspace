#ifndef __REEF_KERNEL_KILL_H__
#define __REEF_KERNEL_KILL_H__

#include "reef/utils.h"

typedef struct CUctx_st* CUcontext; // forward declear cuda stream type

namespace reef {

struct GPUPhysicalMemoryBlock {
    size_t size;
    void* memobj; // cuda internal pointer
    uint32_t rmhandle; // handle for rm api
};

typedef GPUPhysicalMemoryBlock* GPUPMHandle; // handle for GPU physical memory
typedef void* GPUVAHandle; // handle for GPU virtual address


/// @brief A helper class for allocating GPU memory (with physical memory and virtual address),
///        and remapping the virtual/physical memory.
///        We use low-level kernel-driver calls (i.e., ioctl) to directly change 
///        the memory mapping without `ummap` the memory first.
class GPUMemMapHelper {

typedef CUresult (*internalMemAlloc)(CUcontext, size_t , void**, void**);

typedef struct MemObjRMHandles {
    uint32_t struct_size;
    uint32_t reserved0;
    /// RM hMemory
    uint32_t rmMemory;
    uint32_t reserved1;
    void *reserved2;
    size_t reserved3;
    void *reserved4;
    size_t reserved05;
} MemObjRMHandles;

typedef CUresult (*internalGetMemobjRMHandle)(MemObjRMHandles *, CUcontext, void*);

typedef CUresult (*internalGetDeviceUUID) (CUuuid *,CUdevice);

private:
    int nvctlFd_; // fd for /dev/nvidia-ctl
    int nvuvmFd_; // fd for /dev/nvidia-uvm
    int rmClientHandle_; // handle for rmClient

    CUcontext cuCtx_;
    CUuuid uuid_;


    static const uint64_t internalMemAllocIdx;
    internalMemAlloc allocFunc_;

    static const uint64_t internalGetMemobjRMHandleIdx;
    internalGetMemobjRMHandle getMemobjRMHandleFunc_;

    static const uint64_t internalGetDeviceUUIDIdx;
    internalGetDeviceUUID getDeviceUUIDFunc_;
public:
    GPUMemMapHelper(int nvctlFd, int nvuvmFd, int rmClientHandle, CUcontext ctx);

    /// @brief Allocate GPU device memory 
    /// @param vah [out] handle for virtual address
    /// @param pm  [out] handle for physical memory
    /// @param size [in] memory size, must be multiples of 2MB
    /// @return 
    CUresult allocateGPUMemoryPair(GPUVAHandle *vah, GPUPMHandle *pm, size_t size);

    /// @brief Remap the physical page of the virtual address.
    /// @param vah [in] handle for virtual address
    /// @param pmh [in] handlef or physical memory
    /// @return 
    CUresult remapGPUMemory(GPUVAHandle vah, GPUPMHandle pmh);

    // TODO: free the allocated memory

private:
    
};

/// @brief A helper class for flushing gpu instruction cache.
class ICacheFlushHelper {
private:
    int nvctlFd_;
    int rmClientHandle_;
    int rmSubdevHandle_;

public:
    ICacheFlushHelper(int nvctlFd, int rmClientHandle, int rmSubdevHandle);

    CUresult gpuFlushICache();
};

/// @brief A helper class for kill the running kernel.
///        It provides both GPUMemMapHelper hand ICacheFlushHelper; 
class KernelKiller {
public:
    ICacheFlushHelper *iCacheFlushHelper;
    GPUMemMapHelper *gpuMemMapHelper;

private:
    int nvctlFd_;
    int nvuvmFd_;
    int rmClientHandle_;
    int rmSubdevHandle_;
    CUcontext ctx_;

public:
    KernelKiller(CUcontext ctx);
    ~KernelKiller();

private:
    void getRMhandles();
    void getNVfd();
};
}


#endif // __REEF_KERNEL_KILL_H__
