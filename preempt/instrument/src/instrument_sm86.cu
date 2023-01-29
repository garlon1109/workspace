#include <cstring>

#include "nvbit.h"
#include "instrument.h"

#define INSTR_LEN           16
#define BUF_SIZE            2048
#define NVBIT_INJECT_SIZE   0x140

static const uint64_t load_args_instrs[] =
{
    0x00062000ff047b82,
    0x000fc00000000800,
    0x00062100ff057b82,
    0x000fc00000000800,
    0x00062200ff067b82,
    0x000fc00000000800,
    0x00062300ff077b82,
    0x000fc00000000800,
    0x00062400ff087b82,
    0x000fc00000000800,
};

static const uint64_t exit_if_preempt_selective_instrs[] =
{
    0x000003b000007945,
    0x000fe40003800000,
    0x000000ff0800720c,
    0x000fda0003f05270,
    0x000000a000008947,
    0x000fea0003800000,
    0x0000000704007980,
    0x000ea40000100100,
    0x000000ff0000720c,
    0x004fda0003f05270,
    0x0000035000008947,
    0x000fea0003800000,
    0x0000991000007816,
    0x000fc800000000ff,
    0x000000010000780c,
    0x000fda0003f05270,
    0x000000000000094d,
    0x000fea0003800000,
    0x00000002ff007424,
    0x000fe200078e00ff,
    0x0000000804007385,
    0x000fe80000100b06,
    0x0000000704007385,
    0x000fe20000100100,
    0x000000000000794d,
    0x000fea0003800000,
    0x00000000000a7919,
    0x000e220000002200,
    0x0000026000017945,
    0x000fe60003800000,
    0x00000000000b7919,
    0x000e280000002100,
    0x00000000000c7919,
    0x000e280000002300,
    0x0000000000007919,
    0x000e680000002500,
    0x0000000000037919,
    0x000e680000002600,
    0x0000000000097919,
    0x000ea20000002700,
    0x0000000b0cff7212,
    0x001fe2000780fe0a,
    0x0000040000007a24,
    0x002fc800078e0203,
    0x0000050000097a24,
    0x004fca00078e0209,
    0x0000000409087211,
    0x000fca00078218ff,
    0x000000ffff097224,
    0x000fe200008e0605,
    0x000001a000000947,
    0x000fea0003800000,
    0x0000000704007980,
    0x000ea20000100100,
    0x0000013000027945,
    0x000fe20003800000,
    0x000000ff0000720c,
    0x004fda0003f05270,
    0x0000009000008947,
    0x000fea0003800000,
    0x00000001ff037424,
    0x000fe200078e00ff,
    0x0000991000007816,
    0x000fc800000000ff,
    0x0000001708007385,
    0x0001e20000100103,
    0x000000010000780c,
    0x000fda0003f05270,
    0x000000b000000947,
    0x000fea0003800000,
    0x00000002ff007424,
    0x001fe200078e00ff,
    0x0000000804007385,
    0x0001e80000100b06,
    0x0000000704007385,
    0x0001e20000100100,
    0x0000007000007947,
    0x000fea0003800000,
    0x0000001008047980,
    0x000ea40000100b00,
    0x000000060400720c,
    0x004fe20003f06070,
    0x00000001ff047424,
    0x000fe200078e00ff,
    0x00ffffff05057812,
    0x000fc800078ec0ff,
    0x000000070500720c,
    0x000fda0003f06100,
    0x0000001708000385,
    0x0001e80000100104,
    0x0000001008008385,
    0x0001e40000100b06,
    0x0000000000027941,
    0x001fea0003800000,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x00000000fffff984,
    0x000fe20000000800,
    0x0000000000007992,
    0x000fec0000000000,
    0x0000000000017941,
    0x000fea0003800000,
    0xffffffff00007948,
    0x000fe20003800000,
    0x0000000000007b1d,
    0x000fec0000000000,
    0x0000001708087980,
    0x000ea40000100100,
    0x000000ff0800720c,
    0x004fda0003f05270,
    0x0000001000008947,
    0x000fea0003800000,
    0x000000000000794d,
    0x000fea0003800000,
    0x0000000000007941,
    0x000fea0003800000,
};

static void set_nop_instr(char *instr)
{
    *(uint64_t *)(instr + 0) = 0x0000000000007918;
    *(uint64_t *)(instr + 8) = 0x000fc00000000000;
}

static uint16_t get_opcode(const char *instrs)
{
    return *(uint16_t *)instrs;
}

static uint64_t get_jmp_addr(const char *instr)
{
    uint64_t target_addr = *(uint64_t *)(instr + 4);
    target_addr &= 0x0000ffffffffffffULL;
    return target_addr;
}

static void set_jmp_instr(char *instr, const uint64_t jmp_addr)
{
    // for example, jmp 0x7f1234567890
    *(uint64_t *)(instr + 0) = 0x345678900000794a;
    *(uint64_t *)(instr + 8) = 0x000fea0003807f12;
    instr[4] = jmp_addr & 0xff;
    instr[5] = (jmp_addr >> 8) & 0xff;
    instr[6] = (jmp_addr >> 16) & 0xff;
    instr[7] = (jmp_addr >> 24) & 0xff;
    instr[8] = (jmp_addr >> 32) & 0xff;
    instr[9] = (jmp_addr >> 40) & 0xff;
}

static void extract_instrs(const char *instrs, char *load_args_instrs, char *covered_jmp_back_instrs)
{
    int instr_idx = 0;
    for (instr_idx = NVBIT_INJECT_SIZE - INSTR_LEN; instr_idx >= 0; instr_idx -= INSTR_LEN) {
        if (get_opcode(&instrs[instr_idx]) == 0x794a) {
            // JMP instr
            memcpy(covered_jmp_back_instrs, &instrs[instr_idx - INSTR_LEN], 2 * INSTR_LEN);
            break;
        }
    }
    for (instr_idx -= INSTR_LEN; instr_idx >= 0; instr_idx -= INSTR_LEN) {
        if (get_opcode(&instrs[instr_idx]) == 0x7b82) {
            // LDC instr
            memcpy(load_args_instrs, &instrs[instr_idx - 3 * INSTR_LEN], 4 * INSTR_LEN);
            break;
        }
    }
}

static void disable_nvbit_save_context(char *instrs)
{
    set_nop_instr(instrs + 0x00 * INSTR_LEN);
    set_nop_instr(instrs + 0x01 * INSTR_LEN);
    set_nop_instr(instrs + 0x02 * INSTR_LEN);
    set_nop_instr(instrs + 0x03 * INSTR_LEN);
    set_nop_instr(instrs + 0x04 * INSTR_LEN);
    
    set_nop_instr(instrs + 0x0e * INSTR_LEN);
    set_nop_instr(instrs + 0x0f * INSTR_LEN);
    set_nop_instr(instrs + 0x10 * INSTR_LEN);
    set_nop_instr(instrs + 0x11 * INSTR_LEN);
    set_nop_instr(instrs + 0x12 * INSTR_LEN);
}

EXPORT_FUNC void instrument_selective(CUcontext ctx, CUfunction func, CUstream stream)
{
    char instrs[BUF_SIZE];
    char covered_instr[INSTR_LEN];
    CUdeviceptr device_buffer = 0;

    ASSERT_GPU_ERROR(cuMemAllocAsync(&device_buffer, BUF_SIZE, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));

    // save the covered instruction (the first instruction of func)
    uint64_t func_addr = nvbit_get_func_addr(func);
    memcpyDtoHAsync_force(covered_instr, func_addr, device_buffer, INSTR_LEN, stream);
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));

    instrument_nvbit(ctx, func, "exit_if_preempt_selective");

    memcpyDtoHAsync_force(instrs, func_addr, device_buffer, INSTR_LEN, stream);
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
    uint64_t inject_addr = get_jmp_addr(instrs);

    // if you want to dump and see what nvbit instrumented, uncomment the following 5 lines
    // ASSERT_GPU_ERROR(cuMemsetD8Async(device_buffer, 0, BUF_SIZE, stream));
    // memcpyDtoHAsync_force(instrs, inject_addr, device_buffer, BUF_SIZE, stream);
    // ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
    // save_to_file("nvbit.cubin", instrs, BUF_SIZE);
    // return;

    uint32_t tail = 0;

    // load args
    uint32_t fragment_size = sizeof(load_args_instrs);
    memcpy(instrs + tail, load_args_instrs, fragment_size);
    tail += fragment_size;

    // exit_if_preempt logic
    fragment_size = sizeof(exit_if_preempt_selective_instrs);
    memcpy(instrs + tail, exit_if_preempt_selective_instrs, fragment_size);
    tail += fragment_size;

    // execute the covered instruction
    fragment_size = INSTR_LEN;
    memcpy(instrs + tail, covered_instr, fragment_size);
    tail += fragment_size;

    // jump back to the second instruction of func
    fragment_size = INSTR_LEN;
    set_jmp_instr(instrs + tail, func_addr + INSTR_LEN);
    tail += fragment_size;

    assert(tail <= BUF_SIZE);
    memcpyHtoDAsync_force(inject_addr, instrs, device_buffer, tail, stream);
    ASSERT_GPU_ERROR(cuMemFreeAsync(device_buffer, stream));
    ASSERT_GPU_ERROR(cuStreamSynchronize(stream));
}
