#include <iostream>

#include "checker.h"

size_t IdempotenceChecker::idempotent_cnt_ = 0;
size_t IdempotenceChecker::kernel_total_cnt_ = 0;

std::unordered_map<CUstream, IdempotenceChecker> IdempotenceChecker::checkers_;
std::mutex IdempotenceChecker::info_mutex_;
std::unordered_map<int64_t, std::shared_ptr<KernelInfo>> IdempotenceChecker::kernel_infos_;

const std::vector<std::vector<IdempotenceChecker::MemAccessPattern>> IdempotenceChecker::transition_table_ =
{
    //  NONE                        READ                        WRITE
    {   NOT_ACCESSED,               READ_ONLY,                  WRITE_ONLY      },              // NOT_ACCESSED
    {   READ_ONLY,                  READ_ONLY,                  READ_THEN_WRITE },              // READ_ONLY
    {   WRITE_ONLY,                 WRITE_THEN_READ,            WRITE_ONLY      },              // WRITE_ONLY
    {   WRITE_THEN_READ,            WRITE_THEN_READ,            WRITE_THEN_READ_THEN_WRITE},    // WRITE_THEN_READ
    {   READ_THEN_WRITE,            READ_THEN_WRITE,            READ_THEN_WRITE },              // READ_THEN_WRITE
    {   WRITE_THEN_READ_THEN_WRITE, WRITE_THEN_READ_THEN_WRITE, WRITE_THEN_READ_THEN_WRITE},    // WRITE_THEN_READ_THEN_WRITE
};

void IdempotenceChecker::summarize()
{
    for (auto checker : checkers_) checker.second.check();
    printf("summary: idempotent / total: %ld / %ld\n", idempotent_cnt_, kernel_total_cnt_);
}

void IdempotenceChecker::record_mem_access(MemAccessRecord &record)
{
    checkers_[record.stream_].record(record);
}

void IdempotenceChecker::record_kernel_launch(std::shared_ptr<KernelInfo> kernel_info)
{
    info_mutex_.lock();
    kernel_infos_[kernel_info->kernel_idx_] = kernel_info;
    info_mutex_.unlock();
}

IdempotenceChecker::MemAccessPattern IdempotenceChecker::transition(MemAccessPattern pattern, MemAccessType type)
{
    return transition_table_[pattern][type];
}

void IdempotenceChecker::check()
{
    if (kernel_idx_ < 0) return;

    info_mutex_.lock();
    auto info = kernel_infos_[kernel_idx_];
    kernel_infos_.erase(kernel_idx_);
    info_mutex_.unlock();

    kernel_total_cnt_ += 1;
    idempotent_cnt_ += idempotent_;
    std::cout << std::dec << "kernel idx: " << std::setw(6) << kernel_idx_
        << ", idempotent: " << (idempotent_ ? " true" : "false")
        << ", CUfunction: " << std::setw(16) << std::hex << (void *)info->f_
        << ", name: " << info->name_ << ", mangled name: " << info->mangled_name_ << std::endl;
    
    if (!idempotent_) {
        std::cout << std::dec << "kernel idx: " << std::setw(6) << kernel_idx_
            << ", READ_THEN_WRITE on " << std::setw(16) << std::hex << (void *)fail_addr_ 
            << ", memory access cnt:  " << non_idempotent_at_ << " / " << memaccess_cnt_ << " (" 
            << (float)non_idempotent_at_ / (float)memaccess_cnt_ * 100.0 << "%)"
            << std::endl;
    } else {
        
        return;
    }

    std::cout << std::dec << "kernel idx: " << std::setw(6) << kernel_idx_
        << ", grid dim: [" << std::setw(6) << info->gridDimX_ << "," << std::setw(6) << info->gridDimY_ << "," << std::setw(6) << info->gridDimZ_
        << "], block dim: [" << std::setw(6) << info->blockDimX_ << "," << std::setw(6) << info->blockDimY_ << "," << std::setw(6) << info->blockDimZ_
        << "], param list:\n";
    for (auto param : info->params_) {
        param->print();
    }
    std::cout << std::endl;
}

void IdempotenceChecker::reset(int64_t kernel_idx)
{
    idempotent_ = true;
    kernel_idx_ = kernel_idx;
    memaccess_cnt_ = 0;
    non_idempotent_at_ = 0;
    patterns_.clear();
}

void IdempotenceChecker::record(MemAccessRecord &record)
{
    int64_t kernel_idx = record.kernel_idx_;
    if (kernel_idx > kernel_idx_) {
        check();
        reset(kernel_idx);
    }

    for (int i = 0; i < 32; ++i) {
        uint64_t access_addr = record.addrs_[i];
        if (access_addr != 0) memaccess_cnt_ ++;
    }

    if (!idempotent_) return;

    for (int i = 0; i < 32; ++i) {
        uint64_t access_addr = record.addrs_[i];
        if (access_addr == 0) continue;

        const MemAccessPattern pattern = transition(patterns_[access_addr], record.type_);
        patterns_[access_addr] = pattern;
        if (pattern == READ_THEN_WRITE) {
            idempotent_ = false;
            non_idempotent_at_ = memaccess_cnt_;
            fail_addr_ = access_addr;
            return;
        }
    }
}