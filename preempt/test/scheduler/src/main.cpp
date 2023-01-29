#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>

#include "model.h"
#include "reef/reef.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("[ERR] [TEST] model and tensors required, abort...\n");
        exit(-1);
    }

    const std::string modelName(argv[1]);
    const std::string tensorPrefix(argv[2]);
    TRTModel reefModel(modelName + ".onnx", modelName + ".engine");

    cudaStream_t reefStream;
    ASSERT_CUDA_ERROR(cudaStreamCreate(&reefStream));
    
    printf("\n====== reef test for %s ======\n", modelName.c_str());
    // save correct results
    for (auto inputTensor : reefModel.inputTensors()) {
        std::string tensorName = inputTensor.first;
        std::replace(tensorName.begin(), tensorName.end(), '/', '-');
        inputTensor.second->load(tensorPrefix + "_" + tensorName + ".bin");
    }
    printf("\n====== start async infer ======\n" );
    fflush(stdout);
    for (int i = 0; i < 400; i++) {
        reefModel.inferSync(reefStream);
    }

    printf("\n====== start async infer ======\n" );
    fflush(stdout);
    for (int i = 0; i < 1000; i++) {
        reefModel.inferAsync(reefStream);
    }
    ASSERT_CUDA_ERROR(cudaStreamSynchronize(reefStream));

    return 0;
}