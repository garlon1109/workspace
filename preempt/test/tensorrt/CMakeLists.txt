find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

file(GLOB_RECURSE TRT_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_executable(trtinfer ${TRT_SRCS})

target_include_directories(trtinfer
    PRIVATE ${CUDA_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(trtinfer PRIVATE
    ${CUDA_LIBRARIES}
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    TensorRT::nvonnxparser
    cudareef)

install(TARGETS trtinfer RUNTIME DESTINATION ${TEST_BIN_INSTALL_PREFIX}/tensorrt)

set(MMDEPLOY_OPS_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/libmmdeploy_tensorrt_ops.so)
if(EXISTS ${MMDEPLOY_OPS_PATH})
    install(FILES ${MMDEPLOY_OPS_PATH} DESTINATION lib)
endif()

set(TRT_DATA_PATH ${CMAKE_CURRENT_LIST_DIR}/data)
if(EXISTS ${TRT_DATA_PATH})
    install(DIRECTORY ${TRT_DATA_PATH} DESTINATION ${TEST_BIN_INSTALL_PREFIX}/tensorrt)
endif()

set(TRT_MODEL_PATH ${CMAKE_CURRENT_LIST_DIR}/models)
if(EXISTS ${TRT_MODEL_PATH})
    install(DIRECTORY ${TRT_MODEL_PATH} DESTINATION ${TEST_BIN_INSTALL_PREFIX}/tensorrt)
endif()
