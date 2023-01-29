
#ifndef TRT_PREDICT_PLUGIN_RESHAPE_H
#define TRT_PREDICT_PLUGIN_RESHAPE_H

#include <NvInfer.h>
#include <NvUtils.h>
#include <NvInferPlugin.h>

namespace trt_predict
{

// Reshape plugin
class Reshape : public nvinfer1::IPlugin {
 public:
  explicit Reshape(nvinfer1::Dims target_shape);

  Reshape(const void *buf, size_t size);

  int getNbOutputs() const override
  {
    return 1;
  }

  int initialize() override
  {
    return 0;
  }

  void terminate() override
  {
  }

  size_t getWorkspaceSize(int) const override
  {
    return 0;
  }

  int enqueue(int batchSize,
              const void *const *inputs,
              void **outputs,
              void *workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() override
  {
    return sizeof(nvinfer1::Dims) + sizeof(int);
  }

  void serialize(void *buffer) override;

  void configure(const nvinfer1::Dims *,
                 int,
                 const nvinfer1::Dims *,
                 int,
                 int) override
  {
  }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;

 private:
  nvinfer1::Dims shape_;

  int copy_size_;
};

} // end namespace trt_predict

#endif
