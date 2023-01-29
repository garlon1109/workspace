find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

add_executable(scheduler-test src/main.cpp)

target_include_directories(scheduler-test
    PRIVATE ${CUDA_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(scheduler-test PRIVATE
    ${CUDA_LIBRARIES}
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    TensorRT::nvonnxparser
    cudareef)

install(TARGETS scheduler-test RUNTIME DESTINATION ${TEST_BIN_INSTALL_PREFIX}/scheduler)
