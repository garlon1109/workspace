#include <fstream>
#include <iostream>

#include "kernel.h"

KernelParam::KernelParam(void *data, const size_t size)
    : size_(size)
{
    data_ = malloc(size_);
    memcpy(data_, data, size_);
}

KernelParam::~KernelParam()
{
    free(data_);
}

void KernelParam::print()
{
    switch (size_)
    {
    case 1:
    {
        uint8_t *p = (uint8_t *)data_;
        printf("\t    u8[ 0x%x ]\n", *p);
        break;
    }

    case 2:
    {
        uint16_t *p = (uint16_t *)data_;
        printf("\t   u16[ 0x%x ]\n", *p);
        break;
    }

    case 4:
    {
        uint32_t *p = (uint32_t *)data_;
        printf("\t   u32[ 0x%x ]\n", *p);
        break;
    }

    case 8:
    {
        uint64_t *p = (uint64_t *)data_;
        printf("\t   u64[ 0x%lx ]\n", *p);
        break;
    }

    default:
    {
        uint8_t *p = (uint8_t *)data_;
        printf("\t%*ld[ ", 6, size_);
        for (size_t i = 0; i < size_; ++i) {
            printf("%0*x ", 2, p[i]);
            if (i % 8 == 7) printf(", ");
        }
        printf("]\n");
        break;
    }
    }
}

funcGetParamCount KernelInfo::func_get_count_ = nullptr;
funcGetParamInfo KernelInfo::func_get_info_ = nullptr;

void KernelInfo::init()
{
    const void *ppExportTable;
    cuGetExportTable(&ppExportTable, &CU_ETID_ToolsModule);
    func_get_count_ = *(const funcGetParamCount *)((const char *)ppExportTable + funcGetParamCountOffset);
    func_get_info_ = *(const funcGetParamInfo *)((const char *)ppExportTable + funcGetParamInfoOffset);
}

KernelInfo::KernelInfo(CUfunction f,
                       unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                       unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                       void **params, int64_t kernel_idx, const std::string &name, const std::string &mangled_name)
    : f_(f),
      gridDimX_(gridDimX), gridDimY_(gridDimY), gridDimZ_(gridDimZ),
      blockDimX_(blockDimX), blockDimY_(blockDimY), blockDimZ_(blockDimZ),
      kernel_idx_(kernel_idx), name_(name), mangled_name_(mangled_name)
{
    size_t param_cnt;
    func_get_count_(f, &param_cnt);

    for (size_t i = 0; i < param_cnt; ++ i) {
        KernelParamInfo param_info;
        param_info.structSize_ = sizeof(param_info); // for version checking
        func_get_info_(f, i, &param_info);
        params_.emplace_back(std::make_shared<KernelParam>(params[i], param_info.size_));
    }
}