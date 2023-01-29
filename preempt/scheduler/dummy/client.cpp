#include "client.h"
#include "reef/reef.h"
#include "reef/utils.h"

void onContextInit(CUcontext ctx)
{
    UNUSED(ctx);
    RDEBUG("[dummy client]: onContextInit(%p)", ctx);
    return;
}

void onStreamCreate(CUstream stream)
{
    UNUSED(stream);
    RDEBUG("[dummy client]: onStreamCreate(%p)", stream);
    // reef::enablePreemption(stream);
    return;
}

void onStreamDestroy(CUstream stream)
{
    UNUSED(stream);
    RDEBUG("[dummy client]: onStreamDestroy(%p)", stream);
    return;
}

void onNewStreamCommand(CUstream stream)
{
    UNUSED(stream);
    RDEBUG("[dummy client]: onNewStreamCommand(%p)", stream);
    return;
}

void onStreamSynchronized(CUstream stream)
{
    UNUSED(stream);
    RDEBUG("[dummy client]: onStreamSynchronized1(%p)", stream);
    return;
}
