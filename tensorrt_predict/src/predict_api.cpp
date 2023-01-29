#include "trt_predict.h"
#include "common.h"
#include "predict_api.h"

int InferenceCreate(const void *param_bytes,
                 int param_size,
                 int dev_id,
                 PredictorHandle *out)
{
  auto tp = new trt_predict::TensorRTPredict(dev_id);
  if (tp->load_model_by_buffer(param_bytes, param_size) < 0) {
    return -1;
  }
  *out = reinterpret_cast<PredictorHandle>(tp);
  return 0;
}

int PredGetOutputShape(PredictorHandle handle,
                         trt_uint index,
                         trt_uint **shape_data,
                         trt_uint *shape_ndim)
{
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->get_output_shape(index, shape_data, shape_ndim);
}

int PredSetInputAuto(PredictorHandle handle){
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->set_input_auto();
}

int PredSetInput(PredictorHandle handle,
                   const char *key,
                   const trt_float *data,
                   trt_uint size)
{
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->set_input(key, data, size);
}

int PredForward(PredictorHandle handle)
{
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->forward();
}

int PredForwardAsync(PredictorHandle handle, cudaStream_t stream)
{
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->async_forward(stream);
}

int PredGetOutput(PredictorHandle handle,
                    trt_uint index,
                    trt_float *data,
                    trt_uint size)
{
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  return tp->get_output(index, data, size);
}

int PredFree(PredictorHandle handle)
{
  if (handle == nullptr) {
    std::cout << "PredFree: handle is NULL!" << std::endl;
    return 0;
  }
  auto tp = reinterpret_cast<trt_predict::TensorRTPredict *>(handle);
  delete tp;
  return 0;
}
