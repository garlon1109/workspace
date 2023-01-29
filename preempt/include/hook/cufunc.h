#ifndef __HOOK_CUFUNC_H__
#define __HOOK_CUFUNC_H__

#include <cstddef>
#include <cstdint>

struct CUparam_st
{
    const char unknown_0_[0x08];
    const uint32_t offset_;
    const uint32_t size_;
    const char unknown_1_[0x58];
};

struct CUfunc_st
{
    const char unknown_0_[0x008];
    const char *name_;
    const char unknown_1_[0x040];
    const uint32_t instrSize_;
    const char unknown_2_[0x01c];
    const uint64_t entryPointDev_;
    const char unknown_3_[0x1f0];
    const uint32_t paramCnt_;
    const char unknown_4_[0x004];
    const CUparam_st *params_;
    const uint32_t paramTotalSize_;
};

typedef struct KernelParamInfo_st {
    uint32_t structSize_;
    uint32_t unknown_0_;
    /// Size in bytes of the kernel param.
    size_t size_;
    /// Offset in bytes of the kernel param within binary blob of parameters.
    size_t offset_;
    /// bool, nonzero for shared memory parameters.
    size_t isSmemParam_;
    void *unknown_1_;
} KernelParamInfo;

// offset of the functions in the export table
const size_t funcGetParamCountOffset = 0x108;
const size_t funcGetParamInfoOffset = 0x110;

// functions to get kernel param info
typedef CUresult (*funcGetParamCount)(CUfunction, size_t*);
typedef CUresult (*funcGetParamInfo)(CUfunction, size_t, KernelParamInfo*);

#endif
