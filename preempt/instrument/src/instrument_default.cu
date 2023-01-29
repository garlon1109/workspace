#include "instrument.h"

EXPORT_FUNC void instrument_selective(CUcontext ctx, CUfunction func, CUstream stream)
{
    instrument_nvbit(ctx, func, "exit_if_preempt_selective");
}
