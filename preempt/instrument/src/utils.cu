#include <fstream>

#include "nvbit.h"
#include "instrument.h"

void save_to_file(std::string filename, char *buf, size_t size)
{
    std::ofstream out(filename, std::ios::binary);
    out.write(buf, size);
    out.flush();
    out.close();
}

__global__ void memcpyDtoD(void *dst_device, void *src_device, size_t size)
{
    memcpy(dst_device, src_device, size);
}

void memcpyDtoH_force(void *dst_host, const CUdeviceptr src_device, const size_t size, CUstream stream)
{
    CUdeviceptr readable_devptr;
    ASSERT_GPU_ERROR(cuMemAllocAsync(&readable_devptr, size, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
    memcpyDtoD<<<1, 1, 0, stream>>>((void *)readable_devptr, (void *)src_device, size);
    ASSERT_GPU_ERROR(cuMemcpyDtoHAsync_v2(dst_host, readable_devptr, size, stream));
    ASSERT_GPU_ERROR(cuMemFreeAsync(readable_devptr, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
}

void memcpyHtoD_force(CUdeviceptr dst_device, const void *src_host, const size_t size, CUstream stream)
{
    CUdeviceptr readable_devptr;
    ASSERT_GPU_ERROR(cuMemAllocAsync(&readable_devptr, size, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
    ASSERT_GPU_ERROR(cuMemcpyHtoDAsync_v2(readable_devptr, src_host, size, stream));
    memcpyDtoD<<<1, 1, 0, stream>>>((void *)dst_device, (void *)readable_devptr, size);
    ASSERT_GPU_ERROR(cuMemFreeAsync(readable_devptr, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
}

void memcpyDtoHAsync_force(void *dst_host, const CUdeviceptr src_device, const CUdeviceptr device_buffer, const size_t size, CUstream stream)
{
    memcpyDtoD<<<1, 1, 0, stream>>>((void *)device_buffer, (void *)src_device, size);
    ASSERT_GPU_ERROR(cuMemcpyDtoHAsync_v2(dst_host, device_buffer, size, stream));
}

void memcpyHtoDAsync_force(CUdeviceptr dst_device, const void *src_host, const CUdeviceptr device_buffer, const size_t size, CUstream stream)
{
    ASSERT_GPU_ERROR(cuMemcpyHtoDAsync_v2(device_buffer, src_host, size, stream));
    memcpyDtoD<<<1, 1, 0, stream>>>((void *)dst_device, (void *)device_buffer, size);
}
