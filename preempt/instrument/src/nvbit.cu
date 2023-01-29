#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>

#include "nvbit.h"
#include "nvbit_tool.h"
#include "instrument.h"

EXPORT_FUNC std::vector<int> get_args_size(CUfunction func)
{
    return nvbit_get_kernel_argument_sizes(func);
}

void instrument_nvbit(CUcontext ctx, CUfunction func, const char *dev_func_name)
{
    auto instrs = nvbit_get_instrs(ctx, func);
    assert(instrs.size() > 0);
    auto first_instr = instrs.front();

    // instrument the preemption check before the first instruction.
    nvbit_insert_call(first_instr, dev_func_name, IPOINT_BEFORE);
    // set the arg (i.e., preempt_ptr)
    nvbit_add_call_arg_launch_val64(first_instr, 0);
    nvbit_add_call_arg_launch_val64(first_instr, 8);
    nvbit_add_call_arg_launch_val32(first_instr, 16);
    
    nvbit_enable_instrumented(ctx, func, true);
}

EXPORT_FUNC void clear_instrument(CUcontext ctx, CUfunction func)
{
    nvbit_enable_instrumented(ctx, func, false);
}

EXPORT_FUNC void set_instrument_args(CUcontext ctx, CUfunction func, CUdeviceptr preempt_ptr, int64_t kernel_idx, bool idempotent)
{
    assert(kernel_idx < (1ULL << 56));

    char args_buf[24];
    *(uint64_t *)(args_buf +  0) = preempt_ptr;
    *(uint64_t *)(args_buf +  8) = (uint64_t)kernel_idx;
    *(uint32_t *)(args_buf + 16) = (uint32_t)idempotent;
    nvbit_set_at_launch(ctx, func, args_buf, 24);
    nvbit_enable_instrumented(ctx, func, true);
}

void nvbit_at_init()
{
    //printf("NVBIT Init\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus)
{
    return;
}

void nvbit_at_term()
{
    printf("NVBIT Term\n");
}
