#include "instrument.h"

inline __device__ void exit();

// preempt_flag
// 0:   normal mode, no preemption happened
// 1:   preemtion mode, no preemption happened
// 2:   preemtion mode, preemption started

// preempt_ptr:
// |<- unused (7B) ->|<- preempt_flag (1B) ->|<- preempt_idx (8B) ->|<- execute_idx_block_0 (7B) ->|<- preempt_flag_block_0 (1B) ->|<- 8B for block_1 ->|...
INJECT_FUNC void exit_if_preempt_selective(uint64_t *preempt_ptr, uint64_t kernel_idx, uint32_t idempotent)
{
    uint32_t block_idx;
    uint8_t preempt_flag;
    uint8_t *block_preempt_flag_ptr;
    uint8_t *preempt_flag_ptr = ((uint8_t *)preempt_ptr) + 7;

    if (idempotent) {
        preempt_flag = *preempt_flag_ptr;
        if (preempt_flag == 0) goto ret;
        if (preempt_flag == 1) {
            *preempt_flag_ptr = 2;
            preempt_ptr[1] = kernel_idx;
        }
        exit();
    }

    block_idx = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    block_preempt_flag_ptr = ((uint8_t *)preempt_ptr) + 8 * block_idx + 16 + 7;

    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0) goto sync;

    preempt_flag = *preempt_flag_ptr;

    if (preempt_flag == 0) {
        uint64_t *block_ptr = (uint64_t *)(block_preempt_flag_ptr - 7);
        uint64_t block_execute_idx_with_flag = *block_ptr;
        uint64_t block_execute_idx = block_execute_idx_with_flag & 0x00ffffffffffffffULL;
        
        if (block_execute_idx >= kernel_idx) {
            // the block have been executed
            *block_preempt_flag_ptr = 1;
            goto fence;
        }

        // the block have not been executed
        *block_ptr = kernel_idx;
        goto fence;
    } 
    
    *block_preempt_flag_ptr = 1;
    if (preempt_flag == 1) {
        *preempt_flag_ptr = 2;
        preempt_ptr[1] = kernel_idx;
    }

fence:
    __threadfence_block();
sync:
    __syncthreads();

    if (*block_preempt_flag_ptr != 0) exit();

ret:
    return;
}

// preempt_ptr:
// |<- unused (7B) ->|<- preempt_flag (1B) ->|<- preempt_idx (8B) ->|<- execute_idx_block_0 (7B) ->|<- preempt_flag_block_0 (1B) ->|<- 8B for block_1 ->|...
INJECT_FUNC void exit_if_preempt_universal_block_level(uint64_t *preempt_ptr, uint64_t kernel_idx)
{
    uint32_t block_idx = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    
    uint8_t preempt_flag;
    uint8_t *preempt_flag_ptr = ((uint8_t *)preempt_ptr) + 7;
    uint8_t *block_preempt_flag_ptr = ((uint8_t *)preempt_ptr) + 8 * block_idx + 16 + 7;

    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0) goto sync;

    preempt_flag = *preempt_flag_ptr;

    if (preempt_flag == 0) {
        uint64_t *block_ptr = (uint64_t *)(block_preempt_flag_ptr - 7);
        uint64_t block_execute_idx_with_flag = *block_ptr;
        uint64_t block_execute_idx = block_execute_idx_with_flag & 0x00ffffffffffffffULL;
        
        if (block_execute_idx >= kernel_idx) {
            // the block have been executed
            uint8_t block_preempt_flag = block_execute_idx_with_flag >> 56;

            // block_preempt_flag is setted, no need to write
            if (block_preempt_flag != 0) goto sync;

            *block_preempt_flag_ptr = 1;
            goto fence;
        }

        // the block have not been executed
        *block_ptr = kernel_idx;
        goto fence;
    } 
    
    *block_preempt_flag_ptr = 1;
    if (preempt_flag == 1) {
        *preempt_flag_ptr = 2;
        preempt_ptr[1] = kernel_idx;
    }

fence:
    __threadfence_block();
sync:
    __syncthreads();

    if (*block_preempt_flag_ptr != 0) exit();
}

// preempt_ptr:
// |<- executed_idx (7B) ->|<- preempt_flag (1B) ->|<- preempt_idx (8B) ->|<- execute_idx_block_0 (7B) ->|
INJECT_FUNC void exit_if_preempt_universal(uint64_t *preempt_ptr, uint64_t kernel_idx)
{
    uint64_t *executed_idx_ptr = preempt_ptr;
    uint64_t *preempt_idx_ptr = preempt_ptr + 1;
    
    uint8_t preempt_flag;
    uint64_t executed_idx;
    uint64_t executed_idx_with_flag;

    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0) goto sync;
    
    executed_idx_with_flag = *executed_idx_ptr;

    while (true) {
        executed_idx = executed_idx_with_flag & 0x00ffffffffffffffULL;
        if (executed_idx >= kernel_idx) goto sync;

        preempt_flag = (executed_idx_with_flag >> 56) & 0xff;
        
        if (preempt_flag == 0) {
            uint64_t old_val = executed_idx_with_flag;
            executed_idx_with_flag = atomicCAS((unsigned long long *)executed_idx_ptr, old_val, kernel_idx);
            if (executed_idx_with_flag == old_val) goto fence;

        } else if (preempt_flag == 2) {
            goto fence;
        
        } else if (preempt_flag == 1) {
            uint64_t old_val = executed_idx_with_flag;
            uint64_t new_val = executed_idx | (2ULL << 56);
            executed_idx_with_flag = atomicCAS((unsigned long long *)executed_idx_ptr, old_val, new_val);
            if (executed_idx_with_flag == old_val) {
                *preempt_idx_ptr = kernel_idx;
                goto fence;
            }
        }
    }

fence:
    __threadfence_block();
sync:
    __syncthreads();

    executed_idx_with_flag = *executed_idx_ptr;
    preempt_flag = (executed_idx_with_flag >> 56) & 0xff;
    if (preempt_flag == 2) exit();
}

// preempt_ptr:
// |<- unused (7B) ->|<- preempt_flag (1B) ->|<- preempt_idx (8B) ->|<- execute_idx_block_0 (7B) ->|
INJECT_FUNC void exit_if_preempt_idempotent(uint64_t *preempt_ptr, uint64_t kernel_idx)
{
    uint8_t *preempt_flag_ptr = ((uint8_t *)preempt_ptr) + 7;
    uint64_t *preempt_idx_ptr = preempt_ptr + 1;

    uint8_t preempt_flag = *preempt_flag_ptr;

    if (preempt_flag == 0) return;
    if (preempt_flag == 1) {
        *preempt_flag_ptr = 2;
        *preempt_idx_ptr = kernel_idx;
    }
    exit();
}

inline __device__ void exit()
{
    asm("exit;");
}