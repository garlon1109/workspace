
#ifndef TRT_PREDICT_H
#define TRT_PREDICT_H

#include <vector>
#include <map>
#include <string>
#include <NvInfer.h>
#include <NvUtils.h>
#include <NvInferPlugin.h>
#include <vector>
//#include "plugin/plugin_factory.h"

namespace trt_predict
{

/*
 * 使用时，先创建TensorrtPredict对象，然后调用load_model，之后用foward接口进行预测
 */
class TensorRTPredict {
 public:
  /*
   * dev_id，指定当前predict对象运行在哪个gpu device上，dev-id从0开始编号
   */
  explicit TensorRTPredict(int dev_id = 0) : runtime_(nullptr),
                                             engine_(nullptr),
                                             context_(nullptr),
                                             inited_(false),
                                             max_batch_size_(0),
                                             dev_id_(dev_id),
                                             current_batch_size_(0),
                                             num_bindings_(0),
                                             last_error_str_("")
  {
  }

  ~TensorRTPredict();

  /*
   * model_file_name，输入已经生成好的engine文件
   * 要判断返回值，如果返回值小于0，则说明加载失败
   */
  int load_model(const char *model_file_name);

  /*
   * load model from model_buffer
   */
  int load_model_by_buffer(const void *model_buffer, size_t size);

  /*
   * key, input_data_name
   * input_data_ptr, 表示输入数据的指针
   * input_data_size，表示输入数据的字节数。注意，因为输入是float类型，这里的input_data_size只的float的个数，而不是data的字节数
   */
  int set_input(const char *key,
                const float *input_data_ptr,
                int input_data_size);

  int set_input_auto(const int batch_size = 1);
  /*
   *
   */
  int forward();

  int async_forward(cudaStream_t stream);
  /*
   * index, the output index
   * output_data_ptr, 表示输出结果的buffer指针，要求外部传入已经申请好的buffer。forward函数不负责申请内存
   * output_data_size，表示输出结果buffer的大小。不能小于预期的output-size
   * 返回值小于0，表示处理失败。
   */
  int get_output(size_t index, float *output_data_ptr, int output_data_size);

  /*
   * index, the output index
   * return output size = batch * output_dim_size
   * return -1 if index is not valid
   */
  int get_output_size(size_t index);

  /*
   * return internal buffer of output_index' shape, do not release the buffer
   */
  int get_output_shape(size_t index,
                       unsigned int **shape_data,
                       unsigned int *shape_ndim);

  /*
   * set profiler
   */
  void set_profiler(nvinfer1::IProfiler *prof);

  /*
   *
   */
  const char *get_last_error();

 private:
  /*
   *
   */
  void set_last_error(const char *error_str);

  /*
   * member:
   */
  nvinfer1::IRuntime *runtime_;

  nvinfer1::ICudaEngine *engine_;

  nvinfer1::IExecutionContext *context_;

  bool inited_;

  int max_batch_size_;

  int dev_id_;

  int current_batch_size_;

  int num_bindings_;

  // for cuda buffer
  std::vector<void *> cuda_buffer_;

  std::vector<int> dims_size_;

  std::map<std::string, int> input_name_to_bindings_id_;

  std::vector<int> output_index_;

  //PluginFactory plugin_factory_;

  std::vector<std::vector<unsigned int>> dims_shape_;

  std::string last_error_str_;
};

} // namespace trt_predict

#endif
