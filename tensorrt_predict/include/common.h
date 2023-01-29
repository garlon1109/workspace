#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include <NvInfer.h>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>

namespace trt_predict
{

#define CHECK(status)                                             \
do {                                                              \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << status;                      \
    abort();                                                      \
  }                                                               \
} while(0)

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#ifndef TP_CHECK_CUDA
#define TP_CHECK_CUDA(func)                                       \
do {                                                              \
  cudaError_t e = (func);                                         \
  if (e != cudaSuccess)                                           \
  {                                                               \
    std::cout << "CUDA_CALL Failed at: "<< __FILE__ << " line: "  \
      << __LINE__ << " Error: " << cudaGetErrorString(e);         \
    return -1;                                                    \
  }                                                               \
} while(0)
#endif

#ifndef TP_CHECK_CUDA_NORETURN
#define TP_CHECK_CUDA_NORETURN(func)                              \
do {                                                              \
  cudaError_t e = (func);                                         \
  if (e != cudaSuccess)                                           \
  {                                                               \
    std::cout << "CUDA_CALL Failed at: "<< __FILE__ << " line: "  \
      << __LINE__ << " Error: " << cudaGetErrorString(e);         \
    return;                                                       \
  }                                                               \
} while(0)
#endif

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity level) : inter_level_(level)
  {
  }

 //below is for tenssort7
  void log(nvinfer1::ILogger::Severity severity, const char *msg) override
 //below is for tenssort8 
//  void log(nvinfer1::ILogger::Severity severity,  nvinfer1::AsciiChar const* msg) noexcept
  {
    if (severity > inter_level_) {
      return;
    }

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      case Severity::kVERBOSE:
        std::cerr << "VERBOSE: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

 private:
  Severity inter_level_;
};

// Device Context guard
class TPContext {
 public:
  explicit TPContext(int dev_id) : cur_dev_id_(dev_id), old_dev_id_(-1)
  {
    TP_CHECK_CUDA_NORETURN(cudaGetDevice(&old_dev_id_));
    if (cur_dev_id_ != -1 && dev_id != old_dev_id_) {
      TP_CHECK_CUDA_NORETURN(cudaSetDevice(dev_id));
    }
  }

  ~TPContext()
  {
    if (old_dev_id_ != -1 && cur_dev_id_ != old_dev_id_) {
      TP_CHECK_CUDA_NORETURN(cudaSetDevice(old_dev_id_));
    }
  }

 private:
  int old_dev_id_;

  int cur_dev_id_;
};

} // end namespace trt_predict

#endif // _TRT_COMMON_H_
