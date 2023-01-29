#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct CUctx_st *CUcontext;

typedef struct CUstream_st *CUstream;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * called when a new CUDA context is created.
 */
void onContextInit(CUcontext ctx);

/**
 * called when a new CUDA stream is created.
 */
void onStreamCreate(CUstream stream);

/**
 * called when a new CUDA stream is destroyed.
 */
void onStreamDestroy(CUstream stream);

/**
 * called when an empty PreemptableStream gets a new command.
 */
void onNewStreamCommand(CUstream stream);

/**
 * called when an non-empty PreemptableStream completes the last command.
 */
void onStreamSynchronized(CUstream stream);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
