#ifndef __REEF_INSTRUMENT_H__
#define __REEF_INSTRUMENT_H__

#include <list>
#include <mutex>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include "reef/utils.h"
#include "hook/cuda_subset.h"
#include "instrument/mm.h"

// interfaces provided by libinstrument_smxx.so
EXPORT_FUNC std::vector<int> get_args_size(CUfunction func);
EXPORT_FUNC void instrument_selective(CUcontext ctx, CUfunction func, CUstream stream);
EXPORT_FUNC void clear_instrument(CUcontext ctx, CUfunction func);
EXPORT_FUNC void set_instrument_args(CUcontext ctx, CUfunction func, CUdeviceptr preempt_ptr, int64_t kernel_idx, bool idempotent);

class PreemptContext
{
public:
    const CUcontext cuContext_;
    const CUstream preemptStream_;
    const CUstream instrumentStream_;
    const std::shared_ptr<FixedBuffer> defaultPreemptBuffer_;

private:
    bool enabled_ = false;
    std::mutex launchMutex_;
    std::mutex instrumentMutex_;
    std::unordered_set<CUfunction> instrumentedFunctions_;

    static std::mutex contextMutex_;
    static std::unordered_map<CUcontext, std::shared_ptr<PreemptContext>> preemptContexts_;

public:
    PreemptContext(CUcontext cuContext, CUstream preemptStream, CUstream instrumentStream);
    ~PreemptContext();

    static size_t getRequiredSize(size_t blockCnt);
    static std::shared_ptr<PreemptContext> create(CUcontext cuContext);
    static bool isBuiltinStream(CUcontext cuContext, CUstream cuStream);
    static CUresult setDefaultArgsThenLaunch(CUfunction func, CUstream stream, std::function<CUresult()> launch);
    
    void enable();
    void instrument(CUfunction func);
    std::shared_ptr<DeviceBuffer> allocPreemptBuffer(bool useDefault);
    CUresult setArgsThenLaunch(CUfunction func, CUdeviceptr preemptPtr, int64_t kernelIdx, bool idempotent, std::function<CUresult()> launch);

    void resetPreemptFlag(CUdeviceptr preemptPtr);
    void setPreemptFlag(CUdeviceptr preemptPtr, uint8_t preemptFlag);
    uint8_t getPreemptFlag(CUdeviceptr preemptPtr, int64_t &preemptIdx);
    uint8_t getThenResetPreemptFlag(CUdeviceptr preemptPtr, int64_t &preemptIdx);
};

#endif