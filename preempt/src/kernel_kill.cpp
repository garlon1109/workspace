#include "reef/kernel_kill.h"
#include "hook/cutable.h"
#include <sys/ioctl.h>
#include <assert.h>

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);

HOOK_C_API HOOK_DECL_EXPORT  CUresult cuCtxGetDevice(CUdevice *device);

namespace reef {

const uint64_t GPUMemMapHelper::internalMemAllocIdx = 10;
const uint64_t GPUMemMapHelper::internalGetMemobjRMHandleIdx = 3;
const uint64_t GPUMemMapHelper::internalGetDeviceUUIDIdx = 11;

GPUMemMapHelper::GPUMemMapHelper(
    int nvctlFd, int nvuvmFd, int rmClientHandle, CUcontext ctx):
     nvctlFd_(nvctlFd), nvuvmFd_(nvuvmFd), rmClientHandle_(rmClientHandle), cuCtx_(ctx)
{
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(
        &CU_ETID_ToolsMemory, 
        GPUMemMapHelper::internalMemAllocIdx,
        &this->allocFunc_
    ));

    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(
        &CU_ETID_ToolsRm,
        GPUMemMapHelper::internalGetMemobjRMHandleIdx,
        &this->getMemobjRMHandleFunc_
    ));

    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(
        &CU_ETID_ToolsDevice,
        GPUMemMapHelper::internalGetDeviceUUIDIdx,
        &this->getDeviceUUIDFunc_
    ));

    CUdevice dev; // FIXME: use the device of the context or pass a CUdevice to the constructor.
    ASSERT_GPU_ERROR(cuCtxGetDevice(&dev));
    ASSERT_GPU_ERROR(this->getDeviceUUIDFunc_(&this->uuid_, dev));
}



CUresult GPUMemMapHelper::allocateGPUMemoryPair(GPUVAHandle *vah, GPUPMHandle *pm, size_t size)
{
    void* va;
    void* memobj;
    const size_t twoMB = 2048 * 1024;
    if (size % twoMB != 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    // allocate memory pair, returns the virtual memory address and physical memory object.
    CUresult ret = this->allocFunc_(this->cuCtx_, size, &va, &memobj);
    if (ret) return ret;

    // get the RM handle for the memobj.
    MemObjRMHandles handle;
    handle.struct_size = sizeof(handle); // must set the struct size for version checking
    ret = this->getMemobjRMHandleFunc_(&handle, this->cuCtx_, memobj);
    if (ret) return ret;

    *vah = (GPUVAHandle)va;
    GPUPhysicalMemoryBlock *block = new GPUPhysicalMemoryBlock;
    block->size = size;
    block->memobj = memobj;
    block->rmhandle = handle.rmMemory;
    *pm = block;
    return ret;
}

CUresult GPUMemMapHelper::remapGPUMemory(GPUVAHandle vah, GPUPMHandle pmh)
{
    // TODO: where to place these definitions? (since they are only used here)
    typedef struct
    {
        CUuuid gpuUuid;
        NvU32           gpuMappingType;     // UvmGpuMappingType
        NvU32           gpuCachingType;     // UvmGpuCachingType
        NvU32           gpuFormatType;      // UvmGpuFormatType
        NvU32           gpuElementBits;     // UvmGpuFormatElementBits
        NvU32           gpuCompressionType; // UvmGpuCompressionType
    } UvmGpuMappingAttributes;

    #define UVM_MAX_GPUS 32

    typedef struct
    {
        NvU64                   base                            ; // IN
        NvU64                   length                          ; // IN
        NvU64                   offset                          ; // IN
        UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
        NvU64                   gpuAttributesCount              ; // IN
        NvU32                   rmCtrlFd;                                          // IN
        NvU32                   hClient;                                           // IN
        NvU32                   hMemory;                                           // IN

        int               rmStatus;                                          // OUT
    } UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

    #define UVM_MAP_EXTERNAL_ALLOCATION  0x21
    
    UVM_MAP_EXTERNAL_ALLOCATION_PARAMS params;
    params.hClient = this->rmClientHandle_;
    params.base = (NvU64) vah;
    params.offset = 0;
    params.length = pmh->size;
    params.rmCtrlFd = this->nvctlFd_;
    params.gpuAttributesCount = 1; // FIXME: maybe bug in multi-GPU env?
    params.hMemory = pmh->rmhandle;
    params.perGpuAttributes[0] = {0};
    params.perGpuAttributes[0].gpuMappingType = 1;
    params.perGpuAttributes[0].gpuUuid = this->uuid_;

    int ret = ioctl(this->nvuvmFd_, UVM_MAP_EXTERNAL_ALLOCATION, &params);
    assert(ret == 0);
    assert(params.rmStatus == 0);
    return CUDA_SUCCESS;
}

ICacheFlushHelper::ICacheFlushHelper(int nvctlFd, int rmClientHandle, int rmSubdevHandle)
    :nvctlFd_(nvctlFd), rmClientHandle_(rmClientHandle), rmSubdevHandle_(rmSubdevHandle)
{

}


/* Definitions for ICache flush ioctl call */

#define NV2080_CTRL_GPU_REG_OP_READ_32                             (0x00000000)
#define NV2080_CTRL_GPU_REG_OP_WRITE_32                            (0x00000001)

typedef struct
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
} NV2080_CTRL_GPU_REG_OP;

typedef struct
{
    NvU32 flags;
    NvU64 route;
} NV2080_CTRL_GR_ROUTE_INFO;

typedef struct
{
    NvHandle hClientTarget;
    NvHandle hChannelTarget;
    NvU32    bNonTransactional;
    NvU32    reserved00[2];
    NvU32    regOpCount;
    NvP64    regOps;
    NV2080_CTRL_GR_ROUTE_INFO grRouteInfo;
} NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS;

#define CU_PRI_REG_GLOBAL       0x2
#define NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL                         (0x00000000)

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    /* XAPI: the 'params' field below is conceptually a union, but since */
    /*       there are hundreds, it's handle by a custom xapi fn.        */
    /*       So here, just hide 'params' from XAPIGEN.                   */
    // XAPI_FIELD_HIDDEN(
    NvP64  params;
    // );
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

#define NV_ESC_RM_CONTROL 0x2A
#define NV_IOCTL_MAGIC      'F'

static int doApiEscape (int fd, int cmd, size_t size, unsigned long request,
                   void *apiBuffer, NvV32 *apiBufferStatus) {

    return ioctl(fd, request, apiBuffer);
}

#define DO_API_ESCAPE(fd, cmd, type, apiBuffer)             \
    doApiEscape((fd),                                       \
                (cmd),                                      \
                sizeof(type),                               \
                _IOWR(NV_IOCTL_MAGIC, (cmd), type),         \
                &(apiBuffer),                               \
                &((apiBuffer).status))

static void write_flush_reg(NV2080_CTRL_GPU_REG_OP* op) {
    op->regOp = NV2080_CTRL_GPU_REG_OP_WRITE_32;
    op->regValueLo = 1; // Value
    op->regAndNMaskLo = ~0;
    op->regAndNMaskHi = 0;
    op->regType = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL;
    op->regOffset = 0x00419b3c;
}

static void read_flush_reg(NV2080_CTRL_GPU_REG_OP* op) {
    op->regOp = NV2080_CTRL_GPU_REG_OP_READ_32;
    op->regType = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL;
    op->regOffset = 5260092;
}

static void read_ecc_reg(NV2080_CTRL_GPU_REG_OP* op) {
    op->regOp = NV2080_CTRL_GPU_REG_OP_READ_32;
    op->regType = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL;
    op->regOffset = 5246976;
}

static void write_ecc_reg(NV2080_CTRL_GPU_REG_OP* op) {
    op->regOp = NV2080_CTRL_GPU_REG_OP_WRITE_32;
    op->regValueLo = 2418411398; // Value
    op->regAndNMaskLo = ~0;
    // op->regAndNMaskHi = 0;
    op->regType = NV2080_CTRL_GPU_REG_OP_TYPE_GLOBAL;
    op->regOffset = 4296704;
}


CUresult ICacheFlushHelper::gpuFlushICache()
{
    NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS param_list = {0};
    NV2080_CTRL_GPU_REG_OP ops[2] = {0};

    write_ecc_reg(&ops[0]);
    write_flush_reg(&ops[1]);

    param_list.regOpCount = 2;
    param_list.regOps = (NvP64) ops;

    NVOS54_PARAMETERS rmctl_param = {0};
    rmctl_param.hClient = this->rmClientHandle_;
    rmctl_param.hObject = this->rmSubdevHandle_;
    rmctl_param.cmd = 0x20800122;
    rmctl_param.params = (NvP64)&param_list;
    rmctl_param.paramsSize = sizeof(param_list);

    int status = DO_API_ESCAPE(this->nvctlFd_, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS, rmctl_param);
    // if (status != 0) return CUDA_ERROR_OPERATING_SYSTEM;
    // if (rmctl_param.status != 0) return CUDA_ERROR_UNKNOWN;
    return CUDA_SUCCESS;
}


KernelKiller::KernelKiller(CUcontext ctx): ctx_(ctx)
{
    this->getNVfd();
    this->getRMhandles();
    this->gpuMemMapHelper = new GPUMemMapHelper(
        this->nvctlFd_, this->nvuvmFd_, 
        this->rmClientHandle_, this->ctx_
    );
    this->iCacheFlushHelper = new ICacheFlushHelper(
        this->nvctlFd_, this->rmClientHandle_, this->rmSubdevHandle_
    );
}

KernelKiller::~KernelKiller()
{
    delete this->gpuMemMapHelper; // FIXME: where to free the gpu memory?
    delete this->iCacheFlushHelper;
}

/// @brief This is a hack of the export table, which may be difference across different CUDA versions.
const size_t GetContextHandlesFuncIdx = 1; 
struct CUDAContextHandles {
        uint32_t size;
        uint32_t devInst;
        uint32_t subdevInst;
        uint32_t rmClient;
        uint32_t rmDevice;
        uint32_t rmSubDevice;
};
typedef CUresult (*GetContextHandles)(
        CUDAContextHandles *pRmClient,
        CUcontext ctx);
    
void KernelKiller::getRMhandles() {
    GetContextHandles getContextHandleFunc;
    ASSERT_GPU_ERROR(getFuncFromCUDAExportTable(&CU_ETID_ToolsRm, GetContextHandlesFuncIdx, &getContextHandleFunc));
    CUDAContextHandles handles;
    handles.size = sizeof(handles);
    ASSERT_GPU_ERROR(getContextHandleFunc(&handles, this->ctx_));
    this->rmClientHandle_ = handles.rmClient;
    this->rmSubdevHandle_ = handles.rmSubDevice;
}


static std::string get_fd_names(int fd) {
    char filePath[1024];
    std::string path = "/proc/self/fd/" + std::to_string(fd);
    auto size = readlink(path.c_str(), filePath, 1024);
    if (size > 0)
        return std::string(filePath);
    else
        return "";
}

void KernelKiller::getNVfd() {
    this->nvuvmFd_ = -1;
    this->nvctlFd_ = -1;
    for (int i = 0; i < 255; i++) {
        auto fd_name = get_fd_names(i);
        if (fd_name == "/dev/nvidiactl") {
            this->nvctlFd_ = i;
        } else if (fd_name == "/dev/nvidia-uvm") {
            this->nvuvmFd_ = i;
        }
        if (this->nvuvmFd_ != -1 && this->nvctlFd_ != -1)
            return;
    }
    fprintf(stderr, "[ERR] Cannot get open fd for /dev/nvidiactl"); 
    exit(EXIT_FAILURE);
}

}

