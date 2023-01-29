#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "reef.h"

#define SLEEP_TIME 10000000
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
}


int main()
{
    // create cuda streams
    cudaStream_t stream, copyStream;
    ASSERT_GPU_ERROR(cudaStreamCreate(&stream));
    ASSERT_GPU_ERROR(cudaStreamCreate(&copyStream));

    // enable the preemption mechanism for a cuda stream
    reef::RFconfig config {
        .queueSize = 8,
        .batchSize = 2,
        .taskTimeout = -1,
        .preemptLevel = reef::PreemptHostQueue
    };
    reef::enablePreemption(stream, config);

    // prepare the GPU kernel inputs
    int *buffer;
    int *host_buffer = new int[TEST_SIZE];
    ASSERT_GPU_ERROR(cudaMalloc((void **)&buffer, sizeof(int) * TEST_SIZE));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));
    ASSERT_GPU_ERROR(cudaMemsetAsync(buffer, 0, sizeof(int) * TEST_SIZE, stream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));

    // launch some kernels into this stream.
    for (int i = 0; i < TEST_SIZE; i++)
        set_value<<<1, 1, 0, stream>>>(buffer, i);

    // sleep for a while    
    usleep(1000);

    // preempt the stream
    reef::preempt(stream);


    // sleep for a while again   
    sleep(1);

    // check if the execution is preempted
    ASSERT_GPU_ERROR(cudaMemcpyAsync(host_buffer, buffer, sizeof(int) * TEST_SIZE, cudaMemcpyDeviceToHost, copyStream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(copyStream));
    
    print_buffer(host_buffer, TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (host_buffer[i] != i + 1) {
            printf("preempt at: buffer[%d]=%d\n", i, host_buffer[i]);
            break;
        }
    }

    // restore the stream
    reef::restore(stream);
    // wait for the completion of the kernels
    ASSERT_GPU_ERROR(cudaStreamSynchronize(stream));

    // check the outputs again
    ASSERT_GPU_ERROR(cudaMemcpyAsync(host_buffer, buffer, sizeof(int) * TEST_SIZE, cudaMemcpyDeviceToHost, copyStream));
    ASSERT_GPU_ERROR(cudaStreamSynchronize(copyStream));

    print_buffer(host_buffer, TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (host_buffer[i] != i+1) {
            printf("check fail: buffer[%d]=%d\n", i, host_buffer[i]);
            return 1;
        }
    }

    // disable the preemption and release the resources
    reef::disablePreemption(stream);

    cudaFree(buffer);
    delete host_buffer;
    printf("pass check\n");
    return 0;
}

