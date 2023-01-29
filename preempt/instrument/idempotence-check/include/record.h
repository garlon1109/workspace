#pragma once

#include <cstdint>

#include "nvbit.h"

enum MemAccessType
{
    NONE    = 0,
    READ    = 1,
    WRITE   = 2,
};

struct MemAccessRecord
{
    CUstream stream_;
    int64_t kernel_idx_;
    MemAccessType type_;
    uint64_t addrs_[32];
};
