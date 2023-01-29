cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_STANDARD 99)

add_executable(sample_test
  sample_test.cpp
)

add_executable(loadtest 
  loadtest.cpp
  loadtest_thread.cpp
  thread.cpp
)

add_executable(preempt
  preempt.cpp
  loadtest_thread.cpp
  thread.cpp
)

include_directories(${PROJECT_SOURCE_DIR} /home/pilot/jiarun.liu/workspace/libpreempt/include/)

target_link_libraries(sample_test predict)

target_link_libraries(loadtest predict pthread cudareef)
target_link_directories(loadtest PUBLIC /home/pilot/jiarun.liu/workspace/libpreempt/build/)

target_link_libraries(preempt predict pthread cudareef)
target_link_directories(preempt PUBLIC /home/pilot/jiarun.liu/workspace/libpreempt/build/)

