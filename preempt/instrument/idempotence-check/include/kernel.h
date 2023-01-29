#pragma once

#include <map>
#include <list>
#include <memory>
#include <string>
#include <cstring>
#include <cstddef>
#include <cstdint>

#include "nvbit.h"
#include "cufunc.h"
#include "cutable.h"

struct KernelParam
{
    void *data_;
    const size_t size_;

    KernelParam(void *data, const size_t size);
    ~KernelParam();

    void print();
};

struct KernelInfo
{
    const CUfunction f_;
    const unsigned int gridDimX_, gridDimY_, gridDimZ_;
    const unsigned int blockDimX_, blockDimY_, blockDimZ_;
    const int64_t kernel_idx_;
    const std::string name_;
    const std::string mangled_name_;
    std::list<std::shared_ptr<KernelParam>> params_;

    static funcGetParamCount func_get_count_;
    static funcGetParamInfo func_get_info_;
    static void init();

    KernelInfo(CUfunction f,
               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
               void **params, int64_t kernel_idx, const std::string &name, const std::string &mangled_name);
};