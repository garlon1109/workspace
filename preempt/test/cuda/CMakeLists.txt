find_package(CUDA REQUIRED)

file(GLOB_RECURSE PURE_CUDA_SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)

foreach(PURE_CUDA_SRC ${PURE_CUDA_SRCS})

    get_filename_component(PURE_CUDA_TARGET ${PURE_CUDA_SRC} NAME_WE)
    
    cuda_add_executable(${PURE_CUDA_TARGET} ${PURE_CUDA_SRC})

    target_include_directories(${PURE_CUDA_TARGET}
        PRIVATE ${CUDA_INCLUDE_DIRS}
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_link_libraries(${PURE_CUDA_TARGET} ${CUDA_LIBRARIES} cudareef)

    install(TARGETS ${PURE_CUDA_TARGET} RUNTIME DESTINATION ${TEST_BIN_INSTALL_PREFIX}/cuda)

endforeach()
