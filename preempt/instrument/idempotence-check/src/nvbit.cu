#include <mutex>
#include <atomic>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <sys/stat.h>

#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

#include "kernel.h"
#include "checker.h"

#define DUMP_SASS true
#define CHANNEL_SIZE (1l << 20)

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x << std::dec

struct CTXstate
{
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;
};

std::recursive_mutex mutex;

/* map to store context state */
std::unordered_map<CUcontext, std::shared_ptr<CTXstate>> ctx_states;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback = false;
std::atomic_bool initialized(false);

/* kernel launch idx, incremented at every launch */
int64_t kernel_idx = 0;

__global__ void flush_channel(ChannelDev* ch_dev)
{
    /* set a kernel idx = -1 to indicate communication thread that this is the
     * termination flag */
    MemAccessRecord record;
    record.kernel_idx_ = -1;
    ch_dev->push(&record, sizeof(MemAccessRecord));
    /* flush channel */
    ch_dev->flush();
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func)
{
    /* get context state from map */
    auto it = ctx_states.find(ctx);
    assert(it != ctx_states.end());
    auto ctx_state = it->second;

    /* Get related functions of the kernel (device function that can be called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) continue;

#if DUMP_SASS
        std::string func_name(nvbit_get_func_name(ctx, f));
        std::string mangled_name(nvbit_get_func_name(ctx, f, true));
        std::stringstream ss;
        ss << "dumps/" << mangled_name.substr(0, 200) << '_' << HEX(f) << ".asm";
        std::ofstream sass_file(ss.str());

        sass_file << "CUfunction:            " << HEX(f) << std::endl;
        sass_file << "function entry point:  " << HEX(nvbit_get_func_addr(f)) << std::endl;
        sass_file << "function mangled name: " << mangled_name << std::endl;
        sass_file << "function name:         " << func_name << std::endl;
#endif

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto instr : instrs) {

#if DUMP_SASS
            sass_file << "0x" << std::setfill('0') << std::setw(6) << std::hex
                << (uint64_t)instr->getOffset() << "\t\t" << instr->getSass() << std::endl;
#endif

            if (instr->getMemorySpace() != InstrType::MemorySpace::GLOBAL &&
                instr->getMemorySpace() != InstrType::MemorySpace::GLOBAL_TO_SHARED) continue;
            
            MemAccessType type = NONE;
            if (instr->isLoad()) type = READ;
            else if (instr->isStore()) type = WRITE;

            int mref_idx = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t *op = instr->getOperand(i);
                if (op->type != InstrType::OperandType::MREF) continue;
                
                /* insert call to the instrumentation function with its
                 * arguments */
                nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                /* predicate value */
                nvbit_add_call_arg_guard_pred_val(instr);
                /* opcode id */
                nvbit_add_call_arg_const_val32(instr, (int32_t)type);
                /* memory reference 64 bit address */
                nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                /* add "space" for kernel function pointer that will be set
                 * at launch time (64 bit value at offset 0 of the dynamic
                 * arguments)*/
                nvbit_add_call_arg_launch_val64(instr, 0);

                nvbit_add_call_arg_launch_val64(instr, 8);
                /* add pointer to channel_dev*/
                nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
                mref_idx++;
            }
        }
    }
}

void *recv_poll(void* args)
{
    CUcontext ctx = (CUcontext)args;

    mutex.lock();
    auto it = ctx_states.find(ctx);
    assert(it != ctx_states.end());
    auto ctx_state = it->second;
    ChannelHost *ch_host = &ctx_state->channel_host;
    mutex.unlock();
    
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
    bool done = false;
    while (!done) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes <= 0) continue;

        uint32_t num_processed_bytes = 0;
        while (num_processed_bytes < num_recv_bytes) {
            MemAccessRecord *record = (MemAccessRecord *)&recv_buffer[num_processed_bytes];
            if (record->kernel_idx_ == -1) {
                done = true;
                break;
            }

            IdempotenceChecker::record_mem_access(*record);
            num_processed_bytes += sizeof(MemAccessRecord);
        }
    }

    free(recv_buffer);
    return nullptr;
}

void enter_kernel_launch(CUfunction func,
                         unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                         CUstream stream, void **kernel_params)
{
    if (!initialized.load()) return;
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (skip_callback) return;

    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    assert(cudaGetLastError() == cudaSuccess);
    if (ctx == 0) return;

    skip_callback = true;
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    instrument_function_if_needed(ctx, func);

    /* get function name and pc */
    const char* func_name = nvbit_get_func_name(ctx, func);
    const char* mangled_name = nvbit_get_func_name(ctx, func, true);
    auto kernel_info = std::make_shared<KernelInfo>(func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                                    kernel_params, kernel_idx, func_name, mangled_name);

    IdempotenceChecker::record_kernel_launch(kernel_info);

    /* set args at launch time */
    char args[16];
    *(CUstream *)(args + 0) = stream;
    *(int64_t  *)(args + 8) = kernel_idx;
    nvbit_set_at_launch(ctx, func, args, 16);
    /* enable instrumented code to run */
    nvbit_enable_instrumented(ctx, func, true);

    kernel_idx++;
    skip_callback = false;
}

void exit_kernel_launch()
{
    if (!initialized.load()) return;
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (skip_callback) return;
    skip_callback = true;
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
    skip_callback = false;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus)
{
    initialized.store(true);
}

void nvbit_at_init()
{
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    KernelInfo::init();

    mkdir("dumps", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

void nvbit_at_term()
{
    IdempotenceChecker::summarize();
}

void nvbit_at_ctx_init(CUcontext ctx)
{
    mutex.lock();
    auto ctx_state = std::make_shared<CTXstate>();
    assert(ctx_states.find(ctx) == ctx_states.end());
    ctx_states[ctx] = ctx_state;
    
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_states.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_poll, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
    mutex.unlock();
}

void nvbit_at_ctx_term(CUcontext ctx)
{
    mutex.lock();
    skip_callback = true;

    /* get context state from map */
    auto it = ctx_states.find(ctx);
    assert(it != ctx_states.end());
    auto ctx_state = it->second;

    /* flush channel */
    flush_channel<<<1, 1>>>(ctx_state->channel_dev);
    /* Make sure flush of channel is complete */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback = false;
    
    ctx_states.erase(it);
    mutex.unlock();
}
