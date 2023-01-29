#include "loadtest_thread.h"
#include <thread>

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
        "Usage: loadtest <model path> <iterations> <thread_number> <if dump output> <dev_id>\n");
    return -1;
  }

  const char *model_path = argv[1];
  int input_w = 1;
  int input_h = 1;
  //int input_w = atoi(argv[2]);
  //int input_h = atoi(argv[3]);
  int iterations = atoi(argv[2]);
//  std::string image_path = argv[5];

  int thread_num = 1;
  if(argc > 3){
    thread_num = atoi(argv[3]);
  }

  bool flag_dump_output = false;
  if(argc > 4){
    std::string dump_output = argv[4];
    if(dump_output == "true"){
      flag_dump_output = true;
    }
  }

  int dev_id = 0;
  if(argc > 5){
    dev_id = atoi(argv[5]);
  }

  LoadTestThread* threads = new LoadTestThread[thread_num];
	LoadTestMonitorThread monitor_thread(threads, thread_num, iterations);
	monitor_thread.start();
	for (int i = 0; i < thread_num; ++i) {
		//threads[i].SetImagePath(image_path);
		threads[i].SetModel(model_path, input_w, input_h);
		threads[i].SetTestNumber(iterations);
		threads[i].SetDevNumber(dev_id);
		threads[i].SetThreadIndex(i);
		threads[i].Init();
	}

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_num; ++i) {
    threads[i].start();
  }
  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_time - start_time);
  float total_test_time = time_span.count() * 1000;

  monitor_thread.join();

	float total_thread_test_time = 0;
  float total_thread_predict_time = 0;
  int total_result_correct_number = 0;
  for (int i = 0; i < thread_num; ++i) {
    std::cout << "thread " << (i + 1) << std::endl;
    std::cout << ": test number: " << threads[i].GetTestNumber() << std::endl;
    std::cout << ", test time: " << threads[i].GetTestTime() << "ms" << std::endl;
    std::cout << ", total predict time: " << threads[i].GetTotalPredictTime() << "ms" << std::endl;
    std::cout << ", avarage predict time: " << threads[i].GetAvaragePredictTime() << "ms" << std::endl;
    total_thread_test_time += threads[i].GetTestTime();
    total_thread_predict_time += threads[i].GetTotalPredictTime();
    total_result_correct_number += threads[i].GetResultCorrectNumber();
  }
  float avarage_thread_predict_time = total_thread_predict_time / iterations;
  std::cout << "avarage thread test time: " << total_thread_test_time / thread_num << std::endl;
  std::cout << ", total test number: " << iterations << std::endl;
  std::cout << ", total correct number: " << total_result_correct_number << std::endl;
  std::cout << "(" << (double)total_result_correct_number / iterations * 100 << "%)" << std::endl;
  std::cout << ", total test time: " << total_test_time << std::endl;
  std::cout << ", total thread predict time: " << total_thread_predict_time << std::endl;
  std::cout << ", avarage thread predict time: " << avarage_thread_predict_time << std::endl;
  delete[] threads;
  std::cout << "test finished" << std::endl;
}
