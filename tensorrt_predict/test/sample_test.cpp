#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <thread>

#include "predict_api.h"

#include <iostream>
#include <fstream>
#include <sstream>

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

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    printf(
        "Usage: sample_test <model path> <iterations>\n");
    return -1;
  }
  int i = 0;
  const char *model_path = argv[1];
  int iterations = atoi(argv[2]);

  long fsize = 0;
  void *model_buf = ReadFile(const_cast<char *>(model_path), "rb", &fsize);
  long json_size = 0;

  PredictorHandle handle;
  int ret =
      InferenceCreate(model_buf, fsize, 0, &handle);
  free(model_buf);
  std::cout << "PredCreate done: ret: " << ret << std::endl;
  if (ret < 0) return -1;

  uint32_t *shape_data = NULL, shape_ndim = 0;
  ret = PredGetOutputShape(handle, 0, &shape_data, &shape_ndim);
  std::cout << "PredGetOutputShape done: ret: " << ret << std::endl;
  if (ret < 0) return -1;

  cudaStream_t reefStream;
  cudaStreamCreate(&reefStream);


  while(i < iterations){
    ret = PredSetInputAuto(handle);
    std::cout << "PredSetInput done: ret: " << ret << std::endl;
    if (ret < 0) return -1;

    auto start_time = std::chrono::high_resolution_clock::now();
    PredForwardAsync(handle, reefStream);
    
//    ret = PredForward(handle);
    auto async_time = std::chrono::high_resolution_clock::now();    
    cudaStreamSynchronize(reefStream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    auto time_span_async = std::chrono::duration_cast<std::chrono::duration<double>>(async_time - start_time);
    std::cout << "====== total forward time cost: " << time_span.count() * 1000 << "ms ====== async time: " << time_span_async.count() * 1000 << "ms ======" << std::endl;
    std::cout << "PredForward done: ret: " << ret << std::endl;
    if (ret < 0) return -1;

    size_t outputSize0 = 1;
    for (size_t j = 0; j < shape_ndim; j++) {
      outputSize0 *= shape_data[j];
    }
    float *outputBuf0 = (float *)malloc(outputSize0 * sizeof(float));
    ret = PredGetOutput(handle, 0, outputBuf0, outputSize0);
    
    std::cout << "PredGetOutput done: ret: " << ret << std::endl;
    if (ret < 0) return -1;
#if 0
    const char* dir = "../results";
    int state = access(dir, R_OK | W_OK);
    if (state != 0) {
      system("mkdir ../results");
      system("chmod -R 777 ../results");
    }
    std::stringstream ss;
    thread_local static int count = 0;
    ss << dir << "/foward_output" << "_" << ++count << ".txt";
    std::ofstream ofs(ss.str(), std::ios_base::binary);
    ofs << "model_1 output: ";
    for (size_t v = 0; v < outputSize0; v ++){
      ofs << outputBuf0[v] << " ";
    }

#endif
    //for (size_t i = 0; i < outputSize0; i++) {
    //  LOG(INFO) << outputBuf0[i] << " ";
    //}
    free(outputBuf0);
    //delete img_resize;
    i++;
  }
    cudaStreamDestroy(reefStream);
    ret = PredFree(handle);
    std::cout << "PredFree done: ret: " << std::endl;
    if (ret < 0) return -1;
    return 0;
}
