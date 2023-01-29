#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>

#include "model.h"

#ifndef IDEMPOTENCE_CHECK
#include "test.h"
#include "reef/reef.h"
#endif

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
    reefModel.inferSync(reefStream);

#ifndef IDEMPOTENCE_CHECK
    for (auto outputTensor : reefModel.outputTensors()) {
        outputTensor.second->saveCorrect();
    }

    printf("\n====== inference baseline test ======\n");
    warmUp(reefModel, reefStream);
    int64_t disableEnqueueNs;
    int64_t disableNs = inferLatencyNs(reefModel, reefStream, &disableEnqueueNs);
    printf("[RESULT] [TEST] disable preemption: %ldus, enqueue: %ldus\n", disableNs / 1000, disableEnqueueNs / 1000);

    printf("\n====== inference correctness test ======\n");
    printf("[INFO] [TEST] reef::enablePreemption()\n");
    assert(reef::REEF_SUCCESS == reef::enablePreemption((CUstream)reefStream));
    // launch to make sure that all kernels are instrumented
    warmUp(reefModel, reefStream);
    correctnessTest(reefModel, reefStream);

    printf("\n====== inference overhead test ======\n");

    warmUp(reefModel, reefStream);
    int64_t enableEnqueueNs;
    int64_t enableNs = inferLatencyNs(reefModel, reefStream, &enableEnqueueNs);
    printf("[RESULT] [TEST] enable preemption: %ldus, enqueue: %ldus\n", enableNs / 1000, enableEnqueueNs / 1000);
    printf("[RESULT] [TEST] overhead: %ldus, %.2f%%\n", (enableNs - disableNs) / 1000, float(enableNs - disableNs) / float(disableNs) * 100);

    printf("\n====== inference preempt test ======\n");
    preemptTest(reefModel, reefStream);

    printf("\n====== test complete ======\n");
    printf("[INFO] [TEST] reef::disablePreemption()\n");
    assert(reef::REEF_SUCCESS == reef::disablePreemption((CUstream)reefStream));
#endif
    ASSERT_CUDA_ERROR(cudaStreamDestroy(reefStream));
    return 0;
}