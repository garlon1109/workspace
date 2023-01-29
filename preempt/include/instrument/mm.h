#ifndef __INSTRUMENT_MM_H__
#define __INSTRUMENT_MM_H__

#include <list>
#include <cstddef>

#include "hook/cuda_subset.h"

class DeviceBuffer
{
public:
    virtual size_t size() const = 0;
    virtual CUdeviceptr devPtr() const = 0;
    virtual void expand(size_t expectedSize) = 0;
};

class FixedBuffer: public DeviceBuffer
{
private:
    CUdeviceptr devPtr_;
    const size_t size_;
    const CUstream opStream_;

public:
    FixedBuffer(CUstream opStream, size_t size);
    ~FixedBuffer();

    virtual size_t size() const override { return size_; }
    virtual CUdeviceptr devPtr() const override { return devPtr_; }
    virtual void expand(size_t expectedSize) override;
};

class ResizableBuffer: public DeviceBuffer
{
public:
    // alloc by default 2M
    static const size_t defaultSize_ = (1UL << 21);
    // VM size, alloc at most 1G
    static const size_t vmSize_ = (1UL << 30);

private:
    CUdeviceptr devPtr_;
    size_t size_;
    const CUstream opStream_;

    size_t granularity_;
    CUmemAccessDesc rwDesc_;
    CUmemAllocationProp prop_;

    struct AllocationHandle
    {
        size_t size_;
        CUmemGenericAllocationHandle cuHandle_;
        AllocationHandle(size_t size, CUmemGenericAllocationHandle cuHandle)
            : size_(size), cuHandle_(cuHandle) {}
    };

    std::list<AllocationHandle> handles_;

public:
    ResizableBuffer(CUstream opStream, size_t size=defaultSize_);
    ~ResizableBuffer();

    virtual size_t size() const override { return size_; }
    virtual CUdeviceptr devPtr() const override { return devPtr_; }
    virtual void expand(size_t expectedSize) override;
};

class InstrMemManager
{
private:
    struct DevPMBlock
    {
        size_t size_;
        CUdeviceptr va_;
        void *memobj_;      // cuda internal pointer
        uint32_t rmHandle_; // handle for rm api
    };

    struct CUDAContextHandles
    {
        uint32_t size;
        uint32_t devInst;
        uint32_t subdevInst;
        uint32_t rmClient;
        uint32_t rmDevice;
        uint32_t rmSubDevice;
    };

    struct MemObjRMHandles
    {
        uint32_t struct_size;
        uint32_t reserved0;
        uint32_t rmMemory;  // RM hMemory
        uint32_t reserved1;
        void *reserved2;
        size_t reserved3;
        void *reserved4;
        size_t reserved05;
    };

    int nvctlFd_ = -1;      // fd for /dev/nvidia-ctl
    int nvuvmFd_ = -1;      // fd for /dev/nvidia-uvm
    int rmClientHandle_;    // handle for rmClient
    int rmSubdevHandle_;

    CUuuid devUUID_;
    const CUdevice cuDevice_;
    const CUcontext cuContext_;

    size_t size_;
    size_t granularity_;

    CUdeviceptr execInstrVABase_;
    CUdeviceptr mappedInstrVABase_;
    CUdeviceptr preemptInstrVABase_;
    std::list<DevPMBlock *> execInstrPMRegion_;
    std::list<DevPMBlock *> preemptInstrPMRegion_;

    // alloc by default 2M
    static const size_t defaultSize_ = (1UL << 21);
    // VM size, alloc at most 1G
    static const size_t vmSize_ = (1UL << 30);

    static const uint64_t memAllocIdx_ = 10;
    CUresult (*memAllocInternal)(CUcontext, size_t, void**, void**);

    static const uint64_t getDeviceUUIDIdx_ = 11;
    CUresult (*getDeviceUUIDInternal)(CUuuid *, CUdevice);

    /// @brief This is a hack of the export table, which may be difference across different CUDA versions.
    static const uint64_t getContextHandlesIdx_ = 1;
    CUresult (*getContextHandlesInternal)(CUDAContextHandles *, CUcontext);

    static const uint64_t getMemObjRMHandleIdx_ = 3;
    CUresult (*getMemObjRMHandleInternal)(MemObjRMHandles *, CUcontext, void*);

    DevPMBlock *allocDevPMBlock(size_t size);
    void freeDevPMBlock(DevPMBlock *pmBlock);
    void mapVAToPM(CUdeviceptr va, DevPMBlock *pmBlock);

public:
    InstrMemManager(CUdevice cuDevice, CUcontext cuContext);
    ~InstrMemManager();

    void mapToExecInstrs();
    void mapToPreemptInstrs();
    void expandTo(size_t expectedSize);
    void flushInstrCache();
};

#endif // __INSTRUMENT_MM_H__
