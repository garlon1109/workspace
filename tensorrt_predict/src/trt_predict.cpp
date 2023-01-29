//
// 2022.12 liujiarun@didiglobal.com
//

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <sstream>
#include "common.h"
#include "trt_predict.h"

#define SET_LAST_ERROR(info)                           \
do {                                                   \
  std::stringstream ss;                                \
  ss << info;                                          \
  this->set_last_error(ss.str().c_str());              \
} while(0)

namespace trt_predict
{

static Logger gLogger(Logger::Severity::kWARNING);

TensorRTPredict::~TensorRTPredict()
{
  TPContext tp_context(this->dev_id_);
  if (!inited_) {
    return;
  }
  for (auto p : cuda_buffer_) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  cuda_buffer_.clear();

  if (context_ != nullptr) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    engine_->destroy();
  }

  if (runtime_ != nullptr) {
    runtime_->destroy();
  }
  //plugin_factory_.destroyPlugin();
}


int TensorRTPredict::load_model_by_buffer(const void *model_buffer, size_t size)
{
  // set GPU device
  TPContext tp_context(this->dev_id_);

  nvinfer1::ILogger* tmpLogger;
  initLibNvInferPlugins(tmpLogger, "");
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  if (runtime_ == nullptr) {
    SET_LAST_ERROR("Create NvInfer runtime failed!");
    return -1;
  }

  engine_ =
      //runtime_->deserializeCudaEngine(model_buffer, size, &plugin_factory_);
      runtime_->deserializeCudaEngine(model_buffer, size);
  if (engine_ == nullptr) {
    SET_LAST_ERROR("Create NvInfer engine failed!");
    return -1;
  }

  context_ = engine_->createExecutionContext();
  if (context_ == nullptr) {
    SET_LAST_ERROR("Create NvInfer context failed!");
    return -1;
  }

  max_batch_size_ = engine_->getMaxBatchSize();

  // get binding names
  num_bindings_ = engine_->getNbBindings();
  cuda_buffer_.resize(num_bindings_, nullptr);
  dims_size_.resize(num_bindings_);
  dims_shape_.resize(num_bindings_);
  for (int i = 0; i < num_bindings_; ++i) {
    bool is_input = engine_->bindingIsInput(i);
    if (is_input) {
      const char *name = engine_->getBindingName(i);
      std::string bn = std::string(name);
      if (input_name_to_bindings_id_.count(bn) != 0) {
        return -1;
      }
      input_name_to_bindings_id_[bn] = i;
    } else {
      output_index_.push_back(i);
    }

    // alloc cuda memory for all bindings
    // the dims is (c,h,w)
    nvinfer1::Dims data_dim = engine_->getBindingDimensions(i);
    int data_size = 1;
    dims_shape_[i].clear();
    dims_shape_[i].push_back(1);
    for (int s = 0; s < data_dim.nbDims; ++s) {
      data_size *= data_dim.d[s];
      dims_shape_[i].push_back(data_dim.d[s]);
    }
    dims_size_[i] = data_size;

    // create GPU buffer
    int max_size = max_batch_size_ * data_size;
    // alloc buffer need input byte size, so mulitply sizeof(float)
    TP_CHECK_CUDA(cudaMalloc(&cuda_buffer_[i], max_size * sizeof(float)));
  }

  // set init flag
  inited_ = true;

  return 0;
}

int TensorRTPredict::load_model(const char *model_file_name)
{
  struct stat st{};
  if (stat(model_file_name, &st) != 0) {
    SET_LAST_ERROR("Get file status failed!");
    return -1;
  }

  size_t file_size = st.st_size;

  // read engine file
  std::vector<char> model_buffer(file_size);
  FILE *fp = fopen(model_file_name, "rb");
  size_t ret = fread(model_buffer.data(), 1, file_size, fp);
  if (ret != file_size) {
    SET_LAST_ERROR("Wrong read file size than expected: ("
       << ret << " vs " << file_size << ")");
    fclose(fp);
    return -1;
  }
  fclose(fp);

  return load_model_by_buffer(model_buffer.data(), file_size);
}

int TensorRTPredict::set_input_auto(const int batch_size){
  TPContext tp_context(this->dev_id_);

  auto iter = input_name_to_bindings_id_.begin();
  while(iter != input_name_to_bindings_id_.end()){
    std::string key = iter->first;
    int input_id = input_name_to_bindings_id_[key];
    int input_dim_size = batch_size * dims_size_[input_id];
    float* p = (float*)malloc(input_dim_size * sizeof(float));
    TP_CHECK_CUDA(cudaMemcpy(cuda_buffer_[input_id],
                           p, 
                           input_dim_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    iter ++;
  }
  
  current_batch_size_ = batch_size;
  return 0;

}

int TensorRTPredict::set_input(const char *key,
                               const float *input_data_ptr,
                               int input_data_size)
{
  TPContext tp_context(this->dev_id_);
  if (key == nullptr || input_data_ptr == nullptr || input_data_size <= 0) {
    SET_LAST_ERROR("Invalid input parameters!");
    return -1;
  }
  std::string key_str = std::string(key);
  if (input_name_to_bindings_id_.count(key_str) == 0) {
   SET_LAST_ERROR("Key not found!");
    return -1;
  }
  int input_id = input_name_to_bindings_id_[key_str];
  int input_dim_size = dims_size_[input_id];
  // check input data size
  if (input_data_size % input_dim_size != 0) {
    SET_LAST_ERROR("Input data size not mod of dim: ("
      << input_data_size << " vs " << input_dim_size << ")");
    return -1;
  }
  int batch_size = input_data_size / input_dim_size;
  if (batch_size > max_batch_size_) {
    SET_LAST_ERROR("Input data batch exceed maximum: ("
      << batch_size << " vs " << max_batch_size_ << ")");
    return -1;
  }

  current_batch_size_ = batch_size;

  // copy to cuda device, need to multiply sizeof(float)
  TP_CHECK_CUDA(cudaMemcpy(cuda_buffer_[input_id],
                           input_data_ptr,
                           input_data_size * sizeof(float),
                           cudaMemcpyHostToDevice));
  return 0;
}

int TensorRTPredict::get_output_shape(size_t index,
                                      unsigned int **shape_data,
                                      unsigned int *shape_dim)
{
  TPContext tp_context(this->dev_id_);
  if (index >= output_index_.size()) {
    SET_LAST_ERROR("Get output shape index exceed vector size: ("
                       << index << " vs " << output_index_.size() << ")");
    return -1;
  }

  int out_bn_index = output_index_[index];
  auto &out_dim = dims_shape_[out_bn_index];

  if (current_batch_size_ != 0) {
    out_dim[0] = current_batch_size_;
  }
  *shape_data = out_dim.data();
  *shape_dim = out_dim.size();
  return 0;
}

int TensorRTPredict::get_output_size(size_t index)
{
  TPContext tp_context(this->dev_id_);
  if (index >= output_index_.size()) {
    SET_LAST_ERROR("Get output size index exceed vector size: ("
      << index << " vs " << output_index_.size() << ")");
    return -1;
  }
  int out_bn_index = output_index_[index];
  return dims_size_[out_bn_index] * current_batch_size_;
}

int TensorRTPredict::get_output(size_t index,
                                float *output_data_ptr,
                                int output_data_size)
{
  TPContext tp_context(this->dev_id_);
  if (index < 0 || index >= output_index_.size()) {
    return -1;
  }
  if (output_data_ptr == nullptr || output_data_size <= 0) {
    return -1;
  }
  int out_size = this->get_output_size(index);
  if (output_data_size > out_size) {
    
    SET_LAST_ERROR("Output data size exceed maximum: ("
      << output_data_size << " vs " << out_size << ")");
    return -1;
  }

  int out_bn_index = output_index_[index];

  // copy output to buffer
  TP_CHECK_CUDA(cudaMemcpy(output_data_ptr,
                           cuda_buffer_[out_bn_index],
                           output_data_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
  return 0;
}

int TensorRTPredict::forward()
{
  TPContext tp_context(this->dev_id_);
  // execute
  if (!context_->execute(current_batch_size_, cuda_buffer_.data())) {
    SET_LAST_ERROR("Forward failed!");
    return -1;
  }
  return 0;
}

int TensorRTPredict::async_forward(cudaStream_t stream)
{
  TPContext tp_context(this->dev_id_);
  if(!context_->enqueue(current_batch_size_, cuda_buffer_.data(), stream, nullptr)){
    SET_LAST_ERROR("Async forward failed!");
    return -1;
  }
  return 0;
}

void TensorRTPredict::set_profiler(nvinfer1::IProfiler *prof)
{
  TPContext tp_context(this->dev_id_);
  context_->setProfiler(prof);
}

void TensorRTPredict::set_last_error(const char *error_str)
{
  last_error_str_ = std::string(error_str);
}

const char *TensorRTPredict::get_last_error()
{
  return this->last_error_str_.c_str();
}

} // end namespace trt_predict
