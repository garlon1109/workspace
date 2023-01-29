#pragma once

#include <vector>
#include <cstdint>

#include "nvbit.h"

#define EXPORT_FUNC extern "C" __attribute__((visibility("default")))
#define INJECT_FUNC EXPORT_FUNC __device__ __noinline__

#define ASSERT_GPU_ERROR(cmd)\
{\
    CUresult error = cmd;\
    if (error == CUDA_ERROR_DEINITIALIZED) {\
        fprintf(stderr, "[WARN] cuda result %d: cuda driver is shutting down at %s:%d\n", error, __FILE__, __LINE__); \
    } else if (error != CUDA_SUCCESS) {\
        const char* str;\
        cuGetErrorString(error, &str);\
        fprintf(stderr, "[ERR] cuda error %d: %s at %s:%d\n", error, str, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

EXPORT_FUNC std::vector<int> get_args_size(CUfunction func);
EXPORT_FUNC void instrument_selective(CUcontext ctx, CUfunction func, CUstream stream);
EXPORT_FUNC void clear_instrument(CUcontext ctx, CUfunction func);
EXPORT_FUNC void set_instrument_args(CUcontext ctx, CUfunction func, CUdeviceptr preempt_ptr, int64_t kernel_idx, bool idempotent);

void save_to_file(std::string filename, char *buf, size_t size);
void instrument_nvbit(CUcontext ctx, CUfunction func, const char *dev_func_name);
void memcpyDtoH_force(void *dst_host, const CUdeviceptr src_device, const size_t size, CUstream stream);
void memcpyHtoD_force(CUdeviceptr dst_device, const void *src_host, const size_t size, CUstream stream);
void memcpyDtoHAsync_force(void *dst_host, const CUdeviceptr src_device, const CUdeviceptr device_buffer, const size_t size, CUstream stream);
void memcpyHtoDAsync_force(CUdeviceptr dst_device, const void *src_host, const CUdeviceptr device_buffer, const size_t size, CUstream stream);

INJECT_FUNC void exit_if_preempt_selective(uint64_t *preempt_ptr, uint64_t kernel_idx, uint32_t idempotent);
