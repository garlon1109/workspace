#pragma once

#include <ctime>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <NvInfer.h>
#include <cuda_runtime.h>

#define ASSERT_CUDA_ERROR(cmd)\
{\
    cudaError_t error = cmd;\
    if (error != cudaSuccess) {\
        const char *str = cudaGetErrorString(error); \
        fprintf(stderr, "[ERR] [CUDA] error %d: %s at %s:%d\n", error, str, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

void fillRandom(void *buf, size_t size)
{
    srand(time(0));
    char *buf1 = (char *)buf;
    int *buf4 = (int *)buf;
    size_t size4 = size / sizeof(int);
    for (size_t i = 0; i < size4; ++i) buf4[i] = rand();
    for (size_t i = size4 * sizeof(int); i < size; ++i) buf1[i] = rand();
}

class Tensor
{
private:
    void *correctVal_ = nullptr;
    void *hostBuffer_ = nullptr;
    void *deviceBuffer_ = nullptr;
    const size_t size_;
    const nvinfer1::Dims dims_;
    const std::string name_;

public:
    Tensor(size_t size, const nvinfer1::Dims &dims, const std::string &name): size_(size), dims_(dims), name_(name)
    {
        correctVal_ = malloc(size_);
        hostBuffer_ = malloc(size_);
        ASSERT_CUDA_ERROR(cudaMalloc(&deviceBuffer_, size_));
    }

    ~Tensor()
    {
        ASSERT_CUDA_ERROR(cudaFree(deviceBuffer_));
    }

    void *getHostBuffer() { return hostBuffer_; }

    void *getDeviceBuffer() { return deviceBuffer_; }

    void copyToHostAsync(cudaStream_t stream)
    {
        ASSERT_CUDA_ERROR(cudaMemcpyAsync(hostBuffer_, deviceBuffer_, size_, cudaMemcpyDeviceToHost, stream));
    }

    void copyToDeviceAsync(cudaStream_t stream)
    {
        ASSERT_CUDA_ERROR(cudaMemcpyAsync(deviceBuffer_, hostBuffer_, size_, cudaMemcpyHostToDevice, stream));
    }

    void copyTo(void *dst)
    {
        memcpy(dst, hostBuffer_, size_);
    }

    void copyFrom(void *src)
    {
        memcpy(hostBuffer_, src, size_);
    }

    bool compare(const void *buffer)
    {
        return memcmp(hostBuffer_, buffer, size_) == 0;
    }

    void saveCorrect()
    {
        memcpy(correctVal_, hostBuffer_, size_);
    }

    bool checkCorrect()
    {
        return memcmp(correctVal_, hostBuffer_, size_) == 0;
    }

    void load(const std::string &filePath)
    {
        memset(hostBuffer_, 0, size_);
        std::ifstream file(filePath, std::ios::binary | std::ios::ate);

        if (!file.good()) {
            file.close();
            printf("[WARN] [TENSOR] input tensor file %s does not exist, tensor %s will be zeros\n", filePath.c_str(), name_.c_str());
            return;
        }

        std::streamsize size = file.tellg();
        assert(size_ >= size);
        printf("[INFO] [TENSOR] load %s (%ld Bytes) to tensor %s (%ld Bytes)\n", filePath.c_str(), size, name_.c_str(), size_);
        file.seekg(0, std::ios::beg);
        assert(file.read((char *)hostBuffer_, size));
        file.close();
    }

    void save(const std::string &filePath)
    {
        printf("[INFO] [TENSOR] save tensor %s (%ld Bytes) to %s\n", name_.c_str(), size_, filePath.c_str());
        std::ofstream file(filePath, std::ios::binary);
        file.write((const char *)hostBuffer_, size_);
        file.close();
    }
};
