#include <cassert>
#include <sys/ioctl.h>

#include "hook/cutable.h"
#include "reef/utils.h"
#include "reef/reef_hook.h"
#include "instrument/mm.h"

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetDevice(CUdevice *device);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);

FixedBuffer::FixedBuffer(CUstream opStream, size_t size): size_(size), opStream_(opStream)
{
    ASSERT_GPU_ERROR(realMemAllocAsync(&devPtr_, size_, opStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(opStream_));
    ASSERT_GPU_ERROR(realMemsetD8Async(devPtr_, 0, size_, opStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(opStream_));
}

FixedBuffer::~FixedBuffer()
{
    ASSERT_GPU_ERROR(realMemFreeAsync(devPtr_, opStream_));
}

void FixedBuffer::expand(size_t)
{
    RDEBUG("fixed device buffer %p cannot be expanded", (void *)devPtr_);
}

ResizableBuffer::ResizableBuffer(CUstream opStream, size_t size): size_(size), opStream_(opStream)
{
    CUdevice device;
    ASSERT_GPU_ERROR(cuCtxGetDevice(&device));
    prop_ = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .win32HandleMetaData = nullptr,
        .reserved = 0,
    };
    rwDesc_ = {
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    // alloc vm space
    ASSERT_GPU_ERROR(cuMemAddressReserve(&devPtr_, vmSize_, 0, 0, 0));
    ASSERT_GPU_ERROR(cuMemGetAllocationGranularity(&granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    
    // alloc pm space
    CUmemGenericAllocationHandle cuHandle;
    size_ = ROUND_UP(size, granularity_);
    ASSERT_GPU_ERROR(cuMemCreate(&cuHandle, size_, &prop_, 0));
    handles_.emplace_back(size_, cuHandle);

    // map vm to pm
    ASSERT_GPU_ERROR(cuMemMap(devPtr_, size_, 0, cuHandle, 0));
    ASSERT_GPU_ERROR(cuMemSetAccess(devPtr_, size_, &rwDesc_, 1));

    // clear buffer
    ASSERT_GPU_ERROR(realMemsetD8Async(devPtr_, 0, size_, opStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(opStream_));
}

ResizableBuffer::~ResizableBuffer()
{
    ASSERT_GPU_ERROR(cuMemUnmap(devPtr_, size_));
    for (auto handle : handles_) ASSERT_GPU_ERROR(cuMemRelease(handle.cuHandle_));
    ASSERT_GPU_ERROR(cuMemAddressFree(devPtr_, vmSize_));
}

void ResizableBuffer::expand(size_t expectedSize)
{
    if (expectedSize <= size_) return;
    if (expectedSize > vmSize_) {
        RDEBUG("resizable buffer %p cannot be expanded to %ldB: exceeds max size of %ldB", (void *)devPtr_, expectedSize, vmSize_);
        return;
    }
    
    size_t newSize = ROUND_UP(expectedSize, granularity_);
    size_t handleSize = newSize - size_;
    RDEBUG("resize buffer %p from %ldB to %ldB", (void *)devPtr_, size_, newSize);

    // alloc pm space
    CUmemGenericAllocationHandle cuHandle;
    ASSERT_GPU_ERROR(cuMemCreate(&cuHandle, handleSize, &prop_, 0));
    handles_.emplace_back(handleSize, cuHandle);

    // map vm to pm
    ASSERT_GPU_ERROR(cuMemMap(devPtr_ + size_, handleSize, 0, cuHandle, 0));
    ASSERT_GPU_ERROR(cuMemSetAccess(devPtr_ + size_, handleSize, &rwDesc_, 1));

    // clear buffer
    ASSERT_GPU_ERROR(realMemsetD8Async(devPtr_ + size_, 0, handleSize, opStream_));
    ASSERT_GPU_ERROR(realStreamSynchronize(opStream_));

    size_ = newSize;
}

InstrMemManager::InstrMemManager(CUdevice cuDevice, CUcontext cuContext): cuDevice_(cuDevice), cuContext_(cuContext)
{
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(&CU_ETID_ToolsMemory, memAllocIdx_, &memAllocInternal));
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(&CU_ETID_ToolsDevice, getDeviceUUIDIdx_, &getDeviceUUIDInternal));
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(&CU_ETID_ToolsRm, getContextHandlesIdx_, &getContextHandlesInternal));
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(&CU_ETID_ToolsRm, getMemObjRMHandleIdx_, &getMemObjRMHandleInternal));

    ASSERT_GPU_ERROR(getDeviceUUIDInternal(&devUUID_, cuDevice_));

    // get nvctlFd_ and nvuvmFd_
    for (int fd = 0; fd < 255; ++fd) {
        char path[64];
        ssize_t len = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), path, 64 - 1);
        if (len < 0) continue;
        assert(len < 64);
        path[len] = '\0';

        if (nvctlFd_ < 0 && strcmp("/dev/nvidiactl", path) == 0) nvctlFd_ = fd;
        else if (nvuvmFd_ < 0 && strcmp("/dev/nvidia-uvm", path) == 0) nvuvmFd_ = fd;
        
        if (nvctlFd_ >= 0 && nvuvmFd_ >= 0) break;
    }

    assert(nvctlFd_ >= 0);
    assert(nvuvmFd_ >= 0);

    // get rmClientHandle_ and rmSubdevHandle_
    CUDAContextHandles handles;
    handles.size = sizeof(handles);
    ASSERT_GPU_ERROR(getContextHandlesInternal(&handles, cuContext_));
    rmClientHandle_ = handles.rmClient;
    rmSubdevHandle_ = handles.rmSubDevice;

    CUmemAllocationProp prop {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = cuDevice_,
        },
        .win32HandleMetaData = nullptr,
        .reserved = 0,
    };

    // alloc vm space
    ASSERT_GPU_ERROR(cuMemAddressReserve(&execInstrVABase_, vmSize_, 0, 0, 0));
    ASSERT_GPU_ERROR(cuMemAddressReserve(&mappedInstrVABase_, vmSize_, 0, 0, 0));
    ASSERT_GPU_ERROR(cuMemAddressReserve(&preemptInstrVABase_, vmSize_, 0, 0, 0));
    ASSERT_GPU_ERROR(cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // assert granularity == page size (2MB)
    assert(granularity_ == (1UL << 21));
    size_ = ROUND_UP(defaultSize_, granularity_);
    DevPMBlock *execInstrPMBlock = allocDevPMBlock(size_);
    DevPMBlock *preemptInstrPMBlock = allocDevPMBlock(size_);
    execInstrPMRegion_.emplace_back(execInstrPMBlock);
    preemptInstrPMRegion_.emplace_back(preemptInstrPMBlock);

    mapVAToPM(execInstrVABase_, execInstrPMBlock);
    mapVAToPM(preemptInstrVABase_, preemptInstrPMBlock);
    mapToExecInstrs();
}

InstrMemManager::~InstrMemManager()
{

}

InstrMemManager::DevPMBlock *InstrMemManager::allocDevPMBlock(size_t size)
{
    void *va;
    void *memobj;
    assert(size > 0 && (size % granularity_ == 0));

    // allocate memory pair, returns the virtual memory address and physical memory object.
    ASSERT_GPU_ERROR(memAllocInternal(cuContext_, size, &va, &memobj);)

    // get the RM handle for the memobj.
    MemObjRMHandles handle;
    // must set the struct size for version checking
    handle.struct_size = sizeof(handle);
    ASSERT_GPU_ERROR(getMemObjRMHandleInternal(&handle, cuContext_, memobj));

    DevPMBlock *pmBlock = new DevPMBlock;
    pmBlock->size_ = size;
    pmBlock->va_ = (CUdeviceptr)va;
    pmBlock->memobj_ = memobj;
    pmBlock->rmHandle_ = handle.rmMemory;
    return pmBlock;
}

void InstrMemManager::freeDevPMBlock(DevPMBlock *pmBlock)
{

}


/// @brief Use low-level kernel-driver calls (i.e., ioctl) to directly change 
///        the memory mapping without `ummap` the memory first.
void InstrMemManager::mapVAToPM(CUdeviceptr va, DevPMBlock *pmBlock)
{
    #define UVM_MAX_GPUS                    32
    #define UVM_MAP_EXTERNAL_ALLOCATION     0x21

    struct UvmGpuMappingAttributes
    {
        CUuuid          gpuUuid             ;
        NvU32           gpuMappingType      ; // UvmGpuMappingType
        NvU32           gpuCachingType      ; // UvmGpuCachingType
        NvU32           gpuFormatType       ; // UvmGpuFormatType
        NvU32           gpuElementBits      ; // UvmGpuFormatElementBits
        NvU32           gpuCompressionType  ; // UvmGpuCompressionType
    };

    struct UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
    {
        NvU64                   base                            ; // IN
        NvU64                   length                          ; // IN
        NvU64                   offset                          ; // IN
        UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS]  ; // IN
        NvU64                   gpuAttributesCount              ; // IN
        NvU32                   rmCtrlFd                        ; // IN
        NvU32                   hClient                         ; // IN
        NvU32                   hMemory                         ; // IN
        int                     rmStatus                        ; // OUT
    };

    UvmGpuMappingAttributes attr {
        .gpuUuid            = devUUID_,
        .gpuMappingType     = 1,
        .gpuCachingType     = 0,
        .gpuFormatType      = 0,
        .gpuElementBits     = 0,
        .gpuCompressionType = 0,
    };

    UVM_MAP_EXTERNAL_ALLOCATION_PARAMS params;
    params.base                   = (NvU64)va;
    params.length                 = pmBlock->size_;
    params.offset                 = 0;
    params.perGpuAttributes[0]    = attr;
    params.gpuAttributesCount     = 1; // FIXME: maybe bug in multi-GPU env?
    params.rmCtrlFd               = (NvU32)nvctlFd_;
    params.hClient                = (NvU32)rmClientHandle_;
    params.hMemory                = pmBlock->rmHandle_;
    params.rmStatus               = 0;

    assert(ioctl(this->nvuvmFd_, UVM_MAP_EXTERNAL_ALLOCATION, &params) == 0);
    assert(params.rmStatus == 0);
}

void InstrMemManager::mapToExecInstrs()
{

}

void InstrMemManager::mapToPreemptInstrs()
{

}

void InstrMemManager::expandTo(size_t expectedSize)
{

}

void InstrMemManager::flushInstrCache()
{
    /* Definitions for ICache flush ioctl call */
    #define NV2080_CTRL_GPU_REG_OP_READ_32          (0x00000000)
    #define NV2080_CTRL_GPU_REG_OP_WRITE_32         (0x00000001)
    #define NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL      (0x00000000)

    struct NV2080_CTRL_GPU_REG_OP
    {
        NvU8    regOp;
        NvU8    regType;
        NvU8    regStatus;
        NvU8    regQuad;
        NvU32   regGroupMask;
        NvU32   regSubGroupMask;
        NvU32   regOffset;
        NvU32   regValueHi;
        NvU32   regValueLo;
        NvU32   regAndNMaskHi;
        NvU32   regAndNMaskLo;
    };

    struct NV2080_CTRL_GR_ROUTE_INFO
    {
        NvU32 flags;
        NvU64 route;
    };

    struct NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS
    {
        NvHandle hClientTarget;
        NvHandle hChannelTarget;
        NvU32    bNonTransactional;
        NvU32    reserved00[2];
        NvU32    regOpCount;
        NvP64    regOps;
        NV2080_CTRL_GR_ROUTE_INFO grRouteInfo;
    };

    struct NVOS54_PARAMETERS
    {
        NvHandle hClient;
        NvHandle hObject;
        NvV32    cmd;
        /* XAPI: the 'params' field below is conceptually a union, but since */
        /*       there are hundreds, it's handle by a custom xapi fn.        */
        /*       So here, just hide 'params' from XAPIGEN.                   */
        // XAPI_FIELD_HIDDEN(
        NvP64    params;
        // );
        NvU32    paramsSize;
        NvV32    status;
    };
    
    NV2080_CTRL_GPU_REG_OP regOps[2] =
    {
        {
            .regOp              = NV2080_CTRL_GPU_REG_OP_WRITE_32,
            .regType            = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL,
            .regStatus          = 0,
            .regQuad            = 0,
            .regGroupMask       = 0,
            .regSubGroupMask    = 0,
            .regOffset          = 4296704,
            .regValueHi         = 0,
            .regValueLo         = 2418411398,
            .regAndNMaskHi      = 0,
            .regAndNMaskLo      = (NvU32)~0,
        },
        {
            .regOp              = NV2080_CTRL_GPU_REG_OP_WRITE_32,
            .regType            = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL,
            .regStatus          = 0,
            .regQuad            = 0,
            .regGroupMask       = 0,
            .regSubGroupMask    = 0,
            .regOffset          = 0x00419b3c,
            .regValueHi         = 0,
            .regValueLo         = 1,
            .regAndNMaskHi      = 0,
            .regAndNMaskLo      = (NvU32)~0,
        },
    };

    NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS paramList;
    memset(&paramList, 0, sizeof(paramList));
    paramList.regOpCount    = 2;
    paramList.regOps        = (NvP64)regOps;

    NVOS54_PARAMETERS rmctlParam {
        .hClient        = (NvU32)rmClientHandle_,
        .hObject        = (NvU32)rmSubdevHandle_,
        .cmd            = 0x20800122,
        .params         = (NvP64)&paramList,
        .paramsSize     = sizeof(paramList),
        .status         = 0,
    };

    assert(ioctl(nvctlFd_, sizeof(NVOS54_PARAMETERS), &rmctlParam) == 0);
    assert(rmctlParam.status == 0);
}
