#include "loadtest_thread.h"

#include <sstream>
#include <chrono>
#include<thread>
#include "reef/reef.h"

bool LoadTestThread::test_failure_ = false;

void *ReadFile(char *file_path, char *omode, long *pfsize) {
  FILE *fd = fopen(file_path, omode);
  if (!fd) {
    printf("failed open %s\n", file_path);
    return nullptr;
  }
  fseek(fd, 0, SEEK_END);
  long fsize = ftell(fd);
  printf("file size: %ld\n", fsize);
  rewind(fd);
  void *file_buf = malloc(fsize);
  fread(file_buf, sizeof(char), fsize, fd);
  fclose(fd);
  *pfsize = fsize;
  return file_buf;
}

LoadTestThread::LoadTestThread()
    : thread_index_(-1)
    , total_number_(0)
    , test_number_(0)
    , dev_number_(-1)
    , current_number_(0)
    , result_correct_number_(0)
    , success_(true)
    , test_time_(0.0f)
    , total_predict_time_(0.0f)
    , avarage_predict_time_(0.0f) {}

LoadTestThread::~LoadTestThread() {
  int ret = PredFree(handle_);
  std::cout << "PredFree done: ret: " << std::endl;
}

void LoadTestThread::Init() {
  long fsize = 0;
  void *model_buf = ReadFile(const_cast<char *>(model_path_), "rb", &fsize);
  long json_size = 0;

  int ret =
      InferenceCreate(model_buf, fsize, 0, &handle_);

  if(model_2_set_ ){
    int ret_2 = InferenceCreate(model_buf, fsize, 0, &handle_2_);
    if(ret_2 < 0){
      std::cout << "model 2 create failed" << std::endl;
    }
  }
  free(model_buf);
  std::cout << "PredCreate done: ret: " << ret << std::endl;

  ret = PredGetOutputShape(handle_, 0, &shape_data_, &shape_ndim_);
  std::cout << "PredGetOutputShape done: ret: " << ret << std::endl;

  std::cout << "Init finished" << std::endl;
}

void LoadTestThread::execute() {
  std::cout << "load test thread start" << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  while (test_failure_ == false && current_number_ < test_number_) {
    if (!Process()) {
      std::cout << "test failure: predict error occured!" << std::endl;
      test_failure_ = false;
      break;
    }
    current_number_++;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  test_time_ = time_span.count() * 1000;
  if (current_number_ != 0) {
    avarage_predict_time_ = total_predict_time_ / current_number_;
  }
  std::cout << "load test thread finished" << std::endl;
}

bool LoadTestThread::Process() {
  int ret = PredSetInputAuto(handle_);
  std::cout << "PredSetInput done: ret: " << ret << std::endl;
  if (ret < 0) return false;

  cudaStream_t Stream;
  if(model_2_set_ == true){
    cudaStreamCreate(&Stream);
    int ret_2 = PredSetInputAuto(handle_2_);
    if(ret_2 < 0){
      std::cout << "Prediction 2 setinput failed" << std::endl;
      return false;
    }
    auto ret = PredForwardAsync(handle_2_, Stream);
  }
  auto start_time = std::chrono::high_resolution_clock::now();
  if(cudastream_set_ == true){
    if(enable_preempt_){
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ret_reef = reef::preempt((CUstream)cudaStream_, true);
    if(reef::REEF_SUCCESS != ret_reef){
        std::cout << "======preempt failed, ret: " <<  ret_reef << "======" << std::endl;
        return false;
    }else{
        std::cout << "====== preempt success" << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span_preempt = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "====== preempt overhead : " << time_span_preempt.count() * 1000 << "ms ======" << std::endl;
  }

    auto t1 = std::chrono::high_resolution_clock::now();
    ret = PredForwardAsync(handle_, cudaStream_);
    if(ret < 0){
      std::cout << "Async forward failed, ret: " << ret << std::endl;
      return false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span_async = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "====== async forward overhead : " << time_span_async.count() * 1000 << "ms ======" << std::endl;


    if(enable_preempt_){
      auto t1 = std::chrono::high_resolution_clock::now();
      if(!preempt_restore_){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      auto ret_reef = reef::restore((CUstream)cudaStream_);
      if(reef::REEF_SUCCESS != ret_reef){
          std::cout << "====== preempt restore failed, ret: " << ret_reef << "======" << std::endl;
      }else{
          std::cout << "======preempt restore success======" << std::endl;
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      auto time_span_preempt_restore = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      std::cout << "====== preempt restore overhead : " << time_span_preempt_restore.count() * 1000 << "ms ======" << std::endl;
    }

    if(model_2_set_){
      auto t_model2_1 = std::chrono::high_resolution_clock::now();
      cudaStreamSynchronize(Stream);
      auto t_model2_2 = std::chrono::high_resolution_clock::now();
      auto time_span_model2_forward = std::chrono::duration_cast<std::chrono::duration<double>>(t_model2_2 - t_model2_1);
      std::cout << "======model_2 synchronize time cost: " << time_span_model2_forward.count() * 1000 << "ms ======" << std::endl;
    }
    auto t_model1_1 = std::chrono::high_resolution_clock::now();                    
    cudaStreamSynchronize(cudaStream_);                                             
    auto t_model1_2 = std::chrono::high_resolution_clock::now();                    
    auto time_span_model2_forward = std::chrono::duration_cast<std::chrono::duration<double>>(t_model1_2 - t_model1_1);
     std::cout << "======model synchronize time cost: " << time_span_model2_forward.count() * 1000 << "ms ======" << std::endl;
  }else{
    ret = PredForward(handle_);
    if(ret <  0){
      std::cout << "sync forward failed, ret: " << ret << std::endl;
      return false;
    }
  }
  auto end_time = std::chrono::high_resolution_clock::now();


  auto time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);

  auto predict_time = time_span.count() * 1000;
  std::cout << "===========  total forward time cost: " << predict_time << "ms ===========" << std::endl;  

  size_t outputSize0 = 1;
  for (size_t j = 0; j < shape_ndim_; j++) {
    outputSize0 *= shape_data_[j];
  }
  float *outputBuf0 = (float *)malloc(outputSize0 * sizeof(float));
  ret = PredGetOutput(handle_, 0, outputBuf0, outputSize0);
  std::cout << "PredGetOutput done: ret: " << ret << std::endl;
  if (ret < 0) return false;

  if(dump_output_ == true){
    const char* dir = "../results_loadtest";
    int state = access(dir, R_OK | W_OK);
    if (state != 0) {
      system("mkdir ../results_loadtest");
      system("chmod -R 777 ../results_loadtest");
    }
    std::stringstream ss;
    thread_local static int count = 0;
    ss << dir << "/foward_output" << "_" << ++count << ".txt";
    std::ofstream ofs(ss.str(), std::ios_base::binary);
    for (size_t v = 0; v < outputSize0; v ++){
      ofs << outputBuf0[v] << " ";
    }
  }
  free(outputBuf0);
  if (success_) {
    total_predict_time_ += predict_time;
  }
  return success_;
}

void LoadTestThread::SetThreadIndex(int index) {
  thread_index_ = index;
}

void LoadTestThread::SetTestNumber(int number) {
  this->test_number_ = number;
}

int LoadTestThread::GetTestNumber() {
  return this->test_number_;
}

void LoadTestThread::SetDevNumber(int number) {
  this->dev_number_ = number;
}

int LoadTestThread::GetDevNumber() {
  return this->dev_number_;
}

int LoadTestThread::GetCurrentNumber() {
  return current_number_;
}

int LoadTestThread::GetResultCorrectNumber() {
  return result_correct_number_;
}

float LoadTestThread::GetTestTime() {
  return test_time_;
}

float LoadTestThread::GetTotalPredictTime() {
  return total_predict_time_;
}

float LoadTestThread::GetAvaragePredictTime() {
  return avarage_predict_time_;
}

bool LoadTestThread::GetTestSuccess() {
  return success_;
}

void LoadTestThread::SetImagePath(const std::string& image_path) {
  this->image_path_ = image_path;
}

void LoadTestThread::DumpOutput(){
  this->dump_output_ = true;
}

void LoadTestThread::SetModel(const char* model_path, const int input_w, const int input_h) {
  this->input_w_ = input_w;
  this->input_h_ = input_h;
  this->model_path_ = model_path;
}

void LoadTestThread::SetModel2(){
  this->model_2_set_ = true;
}

void LoadTestThread::EnablePreempt(){
 this->enable_preempt_ = true; 
}

void LoadTestThread::DisablePreemptRestore(){
 this->preempt_restore_ = false; 
}

void LoadTestThread::SetCudaStream(cudaStream_t &cudaStream){
  this->cudaStream_ =  cudaStream;
  cudastream_set_ = true;
}
