#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "reef/reef.h"

#define SLEEP_TIME 1000000
#define TEST_SIZE 16

#define ASSERT_GPU_ERROR(cmd)\
{\
    cudaError_t error = cmd;\
    if (error != cudaSuccess) {\
        const char* str;\
        str = cudaGetErrorString(error);\
        fprintf(stderr, "cuda error: %s at %s:%d\n", str, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

void print_buffer(int* buffer, int size)
{
    printf("\n==== buffer begin ====\n");
    for (int i = 0; i < size; i++) {
        printf("buffer[%d]\t= %d\n", i, buffer[i]);
    }
    printf("===== buffer end =====\n\n");
}

// Device level sleep function
__device__ __forceinline__ void csleep(uint64_t clock_count)
{
    if (clock_count == 0) return;
    clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_offset = clock64() - start_clock;
    }
}

__global__ void set_value(int* buffer, int value)
{
    csleep(SLEEP_TIME);
    buffer[value] = value+1;
    printf("device: execute set_value %d\n", value);
}

__global__ void set_value_normal(int* buffer, int value)
{
    csleep(SLEEP_TIME / 2);
    buffer[value] = value+1;
    printf("device: execute set_value_normal %d\n", value);
}

int main()
{
    cudaStream_t stream, normalStream, copyStream;
    ASSERT_GPU_ERROR(cudaStreamCreate(&stream));
    ASSERT_GPU_ERROR(cudaStreamCreate(&normalStream));
    ASSERT_GPU_ERROR(cudaStreamCreate(&copyStream));

    reef::enablePreemption(stream);

    int *buffer;
    ASSERT_GPU_ERROR(cudaMalloc((void **)&buffer, sizeof(int) * TEST_SIZE));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));
    ASSERT_GPU_ERROR(cudaMemsetAsync(buffer, 0, sizeof(int) * TEST_SIZE, stream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));

    int *host_buffer = new int[TEST_SIZE];
    for (int i = 0; i < TEST_SIZE / 2; i++)
        set_value<<<1, 1, 0, stream>>>(buffer, i);
    for (int i = TEST_SIZE / 2; i < TEST_SIZE * 0.75; i++)
        set_value<<<1, 1, 0, normalStream>>>(buffer, i);
    for (int i = TEST_SIZE * 0.75; i < TEST_SIZE; i++)
        set_value_normal<<<1, 1, 0, normalStream>>>(buffer, i);
    
    reef::preempt(stream);

    sleep(1);

    ASSERT_GPU_ERROR(cudaMemcpyAsync(host_buffer, buffer, sizeof(int) * TEST_SIZE, cudaMemcpyDeviceToHost, copyStream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(copyStream));
    
    print_buffer(host_buffer, TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (host_buffer[i] != i + 1) {
            printf("preempt at: buffer[%d]=%d\n", i, host_buffer[i]);
            break;
        }
    }

    reef::restore(stream);
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(normalStream));

    ASSERT_GPU_ERROR(cudaMemcpyAsync(host_buffer, buffer, sizeof(int) * TEST_SIZE, cudaMemcpyDeviceToHost, copyStream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(copyStream));

    print_buffer(host_buffer, TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (host_buffer[i] != i+1) {
            printf("check fail: buffer[%d]=%d\n", i, host_buffer[i]);
            return 1;
        }
    }

    reef::disablePreemption(stream);

    cudaFree(buffer);
    delete host_buffer;
    printf("pass check\n");
    return 0;
}

