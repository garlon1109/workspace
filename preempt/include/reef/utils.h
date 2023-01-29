#ifndef __REEF_UTILS_H__
#define __REEF_UTILS_H__

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

#include "hook/cuda_hook.h"
#include "hook/cuda_subset.h"

#define EXPORT_FUNC extern "C" __attribute__((visibility("default")))
#define UNUSED(expr) do { (void)(expr); } while (0)

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetErrorString(CUresult error, const char **pStr);

#ifdef REEF_DEBUG
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
#else
#define ASSERT_GPU_ERROR(cmd)\
{\
    CUresult error = cmd;\
    if (error != CUDA_SUCCESS && error != CUDA_ERROR_DEINITIALIZED) {\
        const char* str;\
        cuGetErrorString(error, &str);\
        fprintf(stderr, "[ERR] cuda error %d: %s at %s:%d\n", error, str, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}
#endif

#ifdef REEF_DEBUG
#define REEF_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define RDEBUG(format, ...)                                                                     \
    printf("[%s %d:%ld %s:%d %s] " format "\n", "REEF", getpid(), syscall(SYS_gettid), \
            REEF_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__)
#else
#define RDEBUG(format, ...) 
#endif

#define ROUND_UP(X, ALIGN) ((X - 1)/ALIGN + 1)*ALIGN

/* get the page size */
static size_t page_size = sysconf(_SC_PAGESIZE);
static int nullfd = open("/dev/random", O_WRONLY);

inline void *getBase(void *p)
{
    return (void *)((((size_t)p) / page_size) * page_size);
}

inline void *getTop(void *p)
{
    return (void *)((((size_t)p / page_size) + 1) * page_size);
}

inline bool isValid(void *p)
{
    return write(nullfd, p, 1) >= 0;

    // use for debug
    // if (write(nullfd, p, 1) < 0) {
    //     printf("ptr %p is invalid\n", p);
    //     fflush(stdout);
    //     return false;
    // } else {
    //     printf("ptr %p is valid\n", p);
    //     fflush(stdout);
    //     return true;
    // }
}

#endif