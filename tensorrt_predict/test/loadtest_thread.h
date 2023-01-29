#ifndef LOADTEST_SRC_LOADTEST_THREAD_H_
#define LOADTEST_SRC_LOADTEST_THREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "thread.h"
#include "predict_api.h"
//#include "opencv2/opencv.hpp"

class LoadTestThread : public IThread {
 public:
  LoadTestThread();
  ~LoadTestThread();

 public:
  void SetThreadIndex(int index);
  void SetTestNumber(int number);
  int GetTestNumber();
  void SetDevNumber(int number);
  int GetDevNumber();
  int GetCurrentNumber();
  int GetResultCorrectNumber();
  float GetTestTime();
  float GetTotalPredictTime();
  float GetAvaragePredictTime();
  bool GetTestSuccess();
  void SetImagePath(const std::string& image_path);
  void DumpOutput();
  void SetModel(const char* model_path, const int input_w, const int input_h);
  void SetModel2();
  void EnablePreempt();
  void DisablePreemptRestore();
  void SetCudaStream(cudaStream_t &cudaStream);

  void Init();
  bool Process();

 protected:
  virtual void execute();

 private:
  int thread_index_;
  int total_number_;
  int test_number_;
  int dev_number_;
  int current_number_;
  int result_correct_number_;
  float test_time_;
  bool success_;
  float total_predict_time_;
  float avarage_predict_time_;

  int input_w_;
  int input_h_;
  const char* model_path_;
  std::string image_path_;
  //cv::Mat image_processed_;
  bool dump_output_ = false;
  uint32_t* shape_data_ = NULL;
  uint32_t shape_ndim_ = 0;

  PredictorHandle handle_;
   PredictorHandle handle_2_; 
  static bool test_failure_;

  bool model_2_set_ = false;
  
  bool enable_preempt_ = false;
  bool preempt_restore_ = true;

  cudaStream_t cudaStream_;
  bool cudastream_set_ = false;
};

#endif  // LOADTEST_SRC_LOADTEST_THREAD_H_

