#include "cuda_subset.h"

void enter_kernel_launch(CUfunction func,
                         unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                         CUstream stream, void **kernel_params);
void exit_kernel_launch();