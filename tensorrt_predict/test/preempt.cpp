#include "loadtest_thread.h"
#include <thread>
#include "reef/reef.h"

class LoadTestMonitorThread : public IThread {
 public:
  LoadTestMonitorThread(LoadTestThread* threads, int thread_number, int test_number)
      : test_failure_(false) {
    this->threads_ = threads;
    this->thread_number_ = thread_number;
    this->test_number_ = test_number;
  };
  ~LoadTestMonitorThread() {};

 public:
  void execute() {
    while (test_failure_ == false) {
      int current_number = 0;
      bool test_failure = false;
      for (int i = 0; i < thread_number_; i++) {
        current_number += threads_[i].GetCurrentNumber();
        if (threads_[i].GetTestSuccess() == false) {
          test_failure = true;
        }
      }
      if (current_number_ != current_number) {
	      std::cout << "test " << current_number << " / " << test_number_ << std::endl;
        current_number_ = current_number;
      }
      if (current_number_ >= test_number_ || test_failure) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

 private:
  LoadTestThread* threads_;
  int thread_number_;
  bool test_failure_;
  int current_number_;
  int test_number_;
};

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    printf(
        "Usage: loadtest <model path> <iterations> <thread num> <block test1 > <block test2>\n");
    return -1;
  }

  const char *model_path = argv[1];
  int iterations = atoi(argv[2]);
  int thread_num = atoi(argv[3]);

  bool disable_restore = false;
  if(argc > 4 && atoi(argv[4]) != 0){
    disable_restore = true;
  }

  int dev_id = 0;
  reef::RFconfig config {
      .queueSize = 64,
      .batchSize = 16,
      .taskTimeout = -1,
      .preemptLevel = reef::PreemptHostQueue,
  };

  //test0
  std::cout << "====== test -0- ====== Single thread two cudaStream  "  << " ======" << std::endl;
  LoadTestThread* threads = new LoadTestThread[2];
  LoadTestMonitorThread monitor_thread(threads, 2, iterations);
  cudaStream_t cudaStream;
  cudaStreamCreate(&cudaStream);
  auto ret_reef = reef::enablePreemption((CUstream)cudaStream, config);
  if(reef::REEF_SUCCESS != ret_reef){
    std::cout << "enable preemption failed, ret: " <<  ret_reef << std::endl;
  }else{
    std::cout << "======enable preemption success======" << std::endl;
    threads[0].EnablePreempt();
  }

  monitor_thread.start();
  threads[0].SetTestNumber(iterations);
  threads[0].SetDevNumber(dev_id);
  threads[0].SetThreadIndex(0);
  threads[0].SetModel(model_path, 0, 0); 
  if(disable_restore){
    threads[0].DisablePreemptRestore();
  }
  threads[0].SetCudaStream(cudaStream);
  threads[0].SetModel2();
  threads[0].Init();

  threads[1].SetTestNumber(iterations);
  threads[1].SetDevNumber(dev_id);
  threads[1].SetThreadIndex(1);
  threads[1].SetModel(model_path, 0, 0);
  threads[1].SetCudaStream(cudaStream);
  threads[1].SetModel2();

  threads[1].Init();

  auto start_time = std::chrono::high_resolution_clock::now();

  threads[0].start();
  threads[0].join();

  std::cout << "\n====== compare: two stream model forward with no preemption ======\n" << std::endl;

  threads[1].start();
  threads[1].join();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  float total_test_time = time_span.count() * 1000;

  monitor_thread.join();
  delete[] threads;

  ret_reef = reef::disablePreemption((CUstream)cudaStream);             
  if(reef::REEF_SUCCESS != ret_reef){                                               
    std::cout << "disable preemption failed, ret: " <<  ret_reef << std::endl;       
  }else{                                                                            
    std::cout << "disable preemption success" << std::endl;                          
  }                               
  cudaStreamDestroy(cudaStream);
  std::cout << "====== test -0- finished, total test time:  "<< total_test_time <<  " ======"<< std::endl;
  std::cout << "\n\n\n" << std::endl;

  //test1
  std::cout << "====== test -1- ====== Single thread single cudaStream , whether to restore preemption:  "<< !disable_restore  << " ======" << std::endl;
  LoadTestThread* threads_0 = new LoadTestThread[1];
  LoadTestMonitorThread monitor_thread_0(threads_0, 1, iterations);
  cudaStream_t cudaStream_0;
  cudaStreamCreate(&cudaStream_0);
  ret_reef = reef::enablePreemption((CUstream)cudaStream_0, config);
  if(reef::REEF_SUCCESS != ret_reef){
    std::cout << "enable preemption failed, ret: " <<  ret_reef << std::endl;
  }else{
    std::cout << "======enable preemption success======" << std::endl;
    threads_0[0].EnablePreempt();
  }

  monitor_thread_0.start();
  threads_0[0].SetTestNumber(iterations);
  threads_0[0].SetDevNumber(dev_id);
  threads_0[0].SetThreadIndex(0);
  threads_0[0].SetModel(model_path, 0, 0);
  if(disable_restore){
    threads_0[0].DisablePreemptRestore();
  }
  threads_0[0].SetCudaStream(cudaStream_0);
  threads_0[0].Init();

  start_time = std::chrono::high_resolution_clock::now();

  threads_0[0].start();
  threads_0[0].join();

  ret_reef = reef::disablePreemption((CUstream)cudaStream_0);                         
  if(reef::REEF_SUCCESS != ret_reef){                                               
    std::cout << "disable preemption failed, ret: " <<  ret_reef << std::endl;           
  }else{                                                                            
    std::cout << "disable preemption success" << std::endl;                              
  }
  cudaStreamDestroy(cudaStream_0);

  end_time = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  total_test_time = time_span.count() * 1000;

  monitor_thread_0.join();
  delete[] threads_0;

  std::cout << "====== test -1- finished, total test time:  "<< total_test_time <<  " ======"<< std::endl;
  std::cout << "\n\n\n" << std::endl;


  //test2
  std::cout << "====== test -2-  ====== Multi thread, single cudaStream test , whether to restore preemption: "<<  !disable_restore <<" ======" << std::endl;
  LoadTestThread* threads_1 = new LoadTestThread[thread_num];
  LoadTestMonitorThread monitor_thread_1(threads_1, thread_num, iterations);
  cudaStream_t cudaStream_1;
  cudaStreamCreate(&cudaStream_1);
  ret_reef = reef::enablePreemption((CUstream)cudaStream_1, config);
  if(reef::REEF_SUCCESS != ret_reef){                                               
    std::cout << "enable preemption failed, ret: " <<  ret_reef << std::endl;       
  }else{                                                                            
    std::cout << "======enable preemption success======" << std::endl; 
    for(int i =0; i < thread_num; i ++){
       threads_1[i].EnablePreempt(); 
    }
  }
                                          

  monitor_thread_1.start();

  for(int i = 0; i < thread_num; ++i){
    threads_1[i].SetTestNumber(iterations);
    threads_1[i].SetDevNumber(dev_id);
    threads_1[i].SetThreadIndex(i);
    threads_1[i].SetCudaStream(cudaStream_1);
    threads_1[i].SetModel(model_path, 0, 0);
    threads_1[i].Init();
  }
  start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_num; ++i) {
    threads_1[i].start();
  }
  for (int i = 0; i < thread_num; ++i) {
    threads_1[i].join();
  }

  end_time = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  total_test_time = time_span.count() * 1000;

  monitor_thread_1.join();

	float total_thread_test_time = 0;
  float total_thread_predict_time = 0;
  int total_result_correct_number = 0;
  for (int i = 0; i < thread_num; ++i) {
    std::cout << "thread " << (i + 1) << std::endl;
    std::cout << ": test number: " << threads_1[i].GetTestNumber() << std::endl;
    std::cout << ", test time: " << threads_1[i].GetTestTime() << "ms" << std::endl;
    std::cout << ", total predict time: " << threads_1[i].GetTotalPredictTime() << "ms" << std::endl;
    std::cout << ", avarage predict time: " << threads_1[i].GetAvaragePredictTime() << "ms" << std::endl;
    total_thread_test_time += threads_1[i].GetTestTime();
    total_thread_predict_time += threads_1[i].GetTotalPredictTime();
    total_result_correct_number += threads_1[i].GetResultCorrectNumber();
  }
  float avarage_thread_predict_time = total_thread_predict_time / iterations;
  std::cout << "avarage thread test time: " << total_thread_test_time / thread_num << std::endl;
  std::cout << ", total test number: " << iterations << std::endl;
  std::cout << ", total correct number: " << total_result_correct_number << std::endl;
  std::cout << "(" << (double)total_result_correct_number / iterations * 100 << "%)" << std::endl;
  std::cout << ", total test time: " << total_test_time << std::endl;
  std::cout << ", total thread predict time: " << total_thread_predict_time << std::endl;
  std::cout << ", avarage thread predict time: " << avarage_thread_predict_time << std::endl;

  delete[] threads_1;

  ret_reef = reef::disablePreemption((CUstream)cudaStream_1);
  if(reef::REEF_SUCCESS != ret_reef){
    std::cout << "disable preemption failed, ret: " <<  ret_reef << std::endl;
  }else{
    std::cout << "disable preemption success" << std::endl;
  }
  cudaStreamDestroy(cudaStream_1);
  std::cout << "====== test -2- finished, total test time: ======" <<  total_test_time << "======" << std::endl;  
  std::cout << "\n\n\n" << std::endl;


//test3
std::cout << "====== test -3- ====== Multi thread, multi stream test: ======" << std::endl;
  LoadTestThread* threads_2 = new LoadTestThread[thread_num];
  cudaStream_t* cudaStreams_2 = new cudaStream_t[thread_num];
  LoadTestMonitorThread monitor_thread_2(threads_2, thread_num, iterations);
  monitor_thread_2.start();
  int preempt_num = rand() % thread_num;
  std::cout << "====== preempt number: " << preempt_num << " ======" << std::endl;
  for(int i = 0; i < thread_num; i ++){
      cudaStreamCreate(&cudaStreams_2[i]);
  }
  for(int i = 0; i < preempt_num; i++){
    ret_reef = reef::enablePreemption((CUstream)cudaStreams_2[i], config);             
    if(reef::REEF_SUCCESS != ret_reef){                                            
      std::cout << "enable preemption failed, ret: " <<  ret_reef << std::endl;    
    }else{                                                                         
      std::cout << "======enable preemption success======" << std::endl;           
      threads_2[i].EnablePreempt();
      threads_2[i].DisablePreemptRestore();      
    } 
  }

  for(int i = 0; i < thread_num; ++i){
    threads_2[i].SetTestNumber(iterations);
    threads_2[i].SetDevNumber(dev_id);
    threads_2[i].SetThreadIndex(i);
    threads_2[i].SetCudaStream(cudaStreams_2[i]);
    threads_2[i].SetModel(model_path, 0, 0);
    threads_2[i].Init();
  }

  start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_num; ++i) {
    threads_2[i].start();
  }
  for (int i = 0; i < thread_num; ++i) {
    threads_2[i].join();
  }

  end_time = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  total_test_time = time_span.count() * 1000;

  monitor_thread_2.join();

	total_thread_test_time = 0;
  total_thread_predict_time = 0;
  total_result_correct_number = 0;
  for (int i = 0; i < thread_num; ++i) {
    std::cout << "thread " << (i + 1) << std::endl;
    std::cout << ": test number: " << threads_2[i].GetTestNumber() << std::endl;
    std::cout << ", test time: " << threads_2[i].GetTestTime() << "ms" << std::endl;
    std::cout << ", total predict time: " << threads_2[i].GetTotalPredictTime() << "ms" << std::endl;
    std::cout << ", avarage predict time: " << threads_2[i].GetAvaragePredictTime() << "ms" << std::endl;
    total_thread_test_time += threads_2[i].GetTestTime();
    total_thread_predict_time += threads_2[i].GetTotalPredictTime();
    total_result_correct_number += threads_2[i].GetResultCorrectNumber();
  }
  avarage_thread_predict_time = total_thread_predict_time / iterations;
  std::cout << "avarage thread test time: " << total_thread_test_time / thread_num << std::endl;
  std::cout << ", total test number: " << iterations << std::endl;
  std::cout << ", total correct number: " << total_result_correct_number << std::endl;
  std::cout << "(" << (double)total_result_correct_number / iterations * 100 << "%)" << std::endl;
  std::cout << ", total test time: " << total_test_time << std::endl;
  std::cout << ", total thread predict time: " << total_thread_predict_time << std::endl;
  std::cout << ", avarage thread predict time: " << avarage_thread_predict_time << std::endl;

  delete[] threads_2;
  std::cout << "====== test -3- finished, total test time: ======" <<  total_test_time << "======" << std::endl;  
  std::cout << "\n\n\n" << std::endl;


}
