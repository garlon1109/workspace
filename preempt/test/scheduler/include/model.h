#pragma once

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <numeric>
#include <cassert>
#include <fstream>
#include <iostream>

#include <unistd.h>

#include <cuda.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include "tensor.h"

#define DEFAULT_DIM_SIZE 32

uint32_t getSize(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    
    case nvinfer1::DataType::kHALF:
        return 2;

    case nvinfer1::DataType::kINT8:
        return 1;

    case nvinfer1::DataType::kINT32:
        return 4;
    
    case nvinfer1::DataType::kBOOL:
        return 1;
    
    default:
        assert(0);
        return 8;
    }
}

const char *getTypeName(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FLOAT";
    
    case nvinfer1::DataType::kHALF:
        return "HALF";

    case nvinfer1::DataType::kINT8:
        return "INT8";

    case nvinfer1::DataType::kINT32:
        return "INT32";
    
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    
    default:
        assert(0);
        return "UNKNOWN";
    }
}

class TRTLogger: public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

class TRTModel
{
private:
    TRTLogger logger_;
    std::vector<void *> bindings_;
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *ctx_;
    std::map<std::string, std::shared_ptr<Tensor>> inputTensors_;
    std::map<std::string, std::shared_ptr<Tensor>> outputTensors_;

    void buildEngine(const std::string &onnxPath)
    {
        nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger_);
        assert(builder);

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
        assert(network);

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger_);
        assert(parser);
        assert(parser->parseFromFile(onnxPath.c_str(), false));

        nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
        assert(config);

        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        assert(profile);

        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::ITensor *inputTensor = network->getInput(i);
            nvinfer1::Dims dims = inputTensor->getDimensions();
            
            bool dynamicShape = false;
            for (int32_t j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    dims.d[j] = DEFAULT_DIM_SIZE;
                    dynamicShape = true;
                }
            }

            if (dynamicShape) {
                profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
                profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
                profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
            }
        }

        config->addOptimizationProfile(profile);
        engine_ = builder->buildEngineWithConfig(*network, *config);
        printf("[INFO] [MODEL] engine built with model %s\n", onnxPath.c_str());
    }

    void loadEngine(const std::string &enginePath)
    {
        std::ifstream engineFile(enginePath, std::ios::binary | std::ios::ate);
        std::streamsize size = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        assert(engineFile.read(buffer.data(), size));

        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger_);
        assert(runtime);

        bool didInitPlugins = initLibNvInferPlugins(&logger_, "");
        engine_ = runtime->deserializeCudaEngine(buffer.data(), size);
        assert(engine_);

        engineFile.close();
        printf("[INFO] [MODEL] engine %s loaded\n", enginePath.c_str());
    }

    void saveEngine(const std::string &enginePath)
    {
        nvinfer1::IHostMemory *hostMemory = engine_->serialize();

        std::ofstream engineFile(enginePath, std::ios::app);
        engineFile.write((const char *)hostMemory->data(), hostMemory->size());
        engineFile.close();
        printf("[INFO] [MODEL] engine %s saved\n", enginePath.c_str());
    }

    void initContext()
    {
        ctx_ = engine_->createExecutionContext();
        assert(ctx_);

        for (int32_t i = 0; i < engine_->getNbBindings(); i++) {
            bool dynamicShape = false;
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            for (int32_t j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    dims.d[j] = DEFAULT_DIM_SIZE;
                    dynamicShape = true;
                }
            }

            if (dynamicShape && engine_->getNbOptimizationProfiles() > 0) {
                dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
            }

            nvinfer1::DataType dataType = engine_->getBindingDataType(i);
            int64_t volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
            int64_t totalSize = volume * getSize(dataType);

            bool isInput = engine_->bindingIsInput(i);
            std::string tensorName(engine_->getBindingName(i));
            std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(totalSize, dims, tensorName);

            if (isInput) {
                printf("[INFO] [MODEL] created input tensor, name: %s, %s[%d", tensorName.c_str(), getTypeName(dataType), dims.d[0]);
                inputTensors_[tensorName] = tensor;
            } else {
                printf("[INFO] [MODEL] created output tensor, name: %s, %s[%d", tensorName.c_str(), getTypeName(dataType), dims.d[0]);
                outputTensors_[tensorName] = tensor;
            }
            
            for (int32_t j = 1; j < dims.nbDims; ++j) printf(", %d", dims.d[j]);
            printf("] (%ld Bytes)\n", totalSize);

            bindings_.emplace_back(tensor->getDeviceBuffer());
            if (dynamicShape && isInput) ctx_->setBindingDimensions(i, dims);
        }

        printf("[INFO] [MODEL] infer context initialized\n");
    }

public:
    TRTModel(const std::string &onnxPath, const std::string &enginePath)
    {
        if (access(enginePath.c_str(), F_OK) != -1) {
            // the engine file exists
            loadEngine(enginePath);
        } else {
            // otherwise, build and save the engine
            buildEngine(onnxPath);
            saveEngine(enginePath);
        }
        initContext();
        fflush(stdout);
    }

    ~TRTModel() {}

    void inferSync(cudaStream_t stream)
    {
        inferAsync(stream);
        ASSERT_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    void inferAsync(cudaStream_t stream)
    {
        for (auto tensor : inputTensors_) tensor.second->copyToDeviceAsync(stream);
        enqueue(stream);
        for (auto tensor : outputTensors_) tensor.second->copyToHostAsync(stream);
    }

    void enqueue(cudaStream_t stream)
    {
        assert(ctx_->enqueueV2(bindings_.data(), stream, nullptr));
    }

    std::map<std::string, std::shared_ptr<Tensor>> &inputTensors()
    {
        return inputTensors_;
    }

    std::map<std::string, std::shared_ptr<Tensor>> &outputTensors()
    {
        return outputTensors_;
    }
};