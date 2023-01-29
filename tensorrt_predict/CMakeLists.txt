cmake_minimum_required(VERSION 3.5)

project(predict)

if(NOT(DEFINED INSTALL_PATH))
  set(INSTALL_PATH /home/pilot/jiarun.liu)
endif()
message("INSTALL_PATH is ${INSTALL_PATH}")
message("INSTALLED_ROOT is ${INSTALLED_ROOT}")

if(NOT(DEFINED TRT_PATH))
  set(TRT_PATH /home/pilot/jiarun.liu/TensorRT-7.2.1.6)
#  set(TRT_PATH /home/pilot/jiarun.liu/TensorRT-8.4.1.5)
endif()

if(NOT(DEFINED CUDA_PATH))
  set(CUDA_PATH /usr/local/cuda)
endif()

if(NOT(DEFINED OPENCV_PATH))
  set(OPENCV_PATH /home/pilot/jiarun.liu/workspace/thirdparty/opencv-offical)
endif()
message("config types: ${CMAKE_CONFIGURATION_TYPES}")

set(CMAKE_BUILD_TYPE Release)

file(GLOB_RECURSE PROJECT_SRC "src/*")
file(GLOB_RECURSE PROJECT_INC "include/*")

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${CUDA_PATH}/include
                    ${TRT_PATH}/include
		    ${OPENCV_PATH}/include
    )

add_library(${PROJECT_NAME} SHARED ${PROJECT_SRC})

target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_PATH}/lib64/)
target_link_libraries(${PROJECT_NAME} cudart nvinfer nvinfer_plugin)
target_link_directories(${PROJECT_NAME} PUBLIC ${TRT_PATH}/lib)
list(APPEND CMAKE_C_FLAGS ${PLATFORM_FLAGS})
list(APPEND CMAKE_CXX_FLAGS ${PLATFORM_FLAGS} " -std=c++11 -fPIC ")

if (NOT CMAKE_C_FLAGS STREQUAL "")
string(REPLACE ";" " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()

if (NOT CMAKE_CXX_FLAGS STREQUAL "")
string(REPLACE ";" " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

add_library(trt_predict STATIC ${PROJECT_SRC} ${PROJECT_INC})

add_subdirectory(test)
set(OUTPUT_ROOT ${INSTALL_PATH})

set(MY_OUTPUT_ROOT ${INSTALL_PATH}/trt_predict)
install(TARGETS trt_predict
    DESTINATION ${MY_OUTPUT_ROOT}/lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/trt_predict
    DESTINATION ${MY_OUTPUT_ROOT}/include/)
