#pragma once

#include <mutex>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "record.h"
#include "kernel.h"

class IdempotenceChecker
{
private:
    enum MemAccessPattern
    {
        NOT_ACCESSED                = 0,
        READ_ONLY                   = 1,
        WRITE_ONLY                  = 2,
        WRITE_THEN_READ             = 3,
        READ_THEN_WRITE             = 4, // non-idempotent
        WRITE_THEN_READ_THEN_WRITE  = 5,
    };

    const static std::vector<std::vector<MemAccessPattern>> transition_table_;

    static size_t idempotent_cnt_;
    static size_t kernel_total_cnt_;
    
    // CUstream => IdempotenceChecker
    static std::unordered_map<CUstream, IdempotenceChecker> checkers_;

    static std::mutex info_mutex_;
    // kernel idx => KernelInfo
    static std::unordered_map<int64_t, std::shared_ptr<KernelInfo>> kernel_infos_;

    bool idempotent_ = true;
    int64_t non_idempotent_at_ = 0;
    uint64_t fail_addr_ = 0;
    int64_t kernel_idx_ = -1;
    int64_t memaccess_cnt_ = 0;
    std::unordered_map<uint64_t, MemAccessPattern> patterns_;

public:
    IdempotenceChecker() {}
    ~IdempotenceChecker() {}

    static void summarize();
    static void record_mem_access(MemAccessRecord &record);
    static void record_kernel_launch(std::shared_ptr<KernelInfo> kernel_info);
    static MemAccessPattern transition(MemAccessPattern pattern, MemAccessType type);

    void check();
    void reset(int64_t kernel_idx);
    void record(MemAccessRecord &record);
};