# trt_predict

# 简介
本工程主要针对tenssort升级和应用进行多线程多进程测试

# 使用
1.loadtest分支

2.在$PROJECT_ROOT目录下：mkdir build && cd build && cmake .. && make -j8；（使用不同版本的Tenssorrt需要注意修改CMakeList.txt中的TENSORRT_PATH地址)
3.准备测试模型

4.运行scripts下的脚本（通过thread_list指定测试线程数）
