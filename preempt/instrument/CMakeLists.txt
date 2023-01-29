find_package(CUDA REQUIRED)

if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/lib/libnvbit_${CMAKE_HOST_SYSTEM_PROCESSOR}.a)
    message(FATAL_ERROR "Error: cpu arch ${CMAKE_HOST_SYSTEM_PROCESSOR} not supported!")
endif()

if(NOT DEFINED GPU_GEN_CODE)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        RESULT_VARIABLE NVIDIA_SMI_STATUS
        OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NVIDIA_SMI_STATUS EQUAL 0)
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)" "\\1\\2" GPU_GEN_CODE ${CUDA_COMPUTE_CAPABILITY})
    else()
        message(WARNING "nvidia-smi failed with ${NVIDIA_SMI_STATUS}")
        message(FATAL_ERROR "Error: gpu generation code not set! Pass by -DGPU_GEN_CODE=xx")
    endif()
endif()
message(STATUS "GPU_GEN_CODE: ${GPU_GEN_CODE}")

set(ARCH_RELATED_INSTRUMENT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/instrument_sm${GPU_GEN_CODE}.cu
)

if(NOT EXISTS ${ARCH_RELATED_INSTRUMENT_SRC})
    message(WARNING "Warning: Binary-level optimization for gpu arch sm${GPU_GEN_CODE} is not supported.\n"
                    "Preempt level PreemptHostQueue will be default to reduce execution overhead.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEFAULT_PREEMPT_LEVEL_HOST_QUEUE" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEFAULT_PREEMPT_LEVEL_HOST_QUEUE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEFAULT_PREEMPT_LEVEL_HOST_QUEUE" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEFAULT_PREEMPT_LEVEL_HOST_QUEUE")
    set(ARCH_RELATED_INSTRUMENT_SRC
        ${CMAKE_CURRENT_LIST_DIR}/src/instrument_default.cu
    )
endif()

set(INSTRUMENT_INCLUDE_DIR
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(INSTRUMENT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/nvbit.cu
    ${ARCH_RELATED_INSTRUMENT_SRC}
)

set(INJECT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/inject.cu
)

set(NVBIT_LIB
    ${CMAKE_CURRENT_LIST_DIR}/lib/libnvbit_${CMAKE_HOST_SYSTEM_PROCESSOR}.a
)

set(INSTRUMENT_COMMON_FLAGS
    -c
    -O3
    -D_FORCE_INLINES
    -Xcompiler -fPIC
    -Xcompiler -Wall
    -arch=sm_${GPU_GEN_CODE}
    -I ${INSTRUMENT_INCLUDE_DIR}
)

cuda_compile(INSTRUMENT_O
    ${INSTRUMENT_SRC}
    OPTIONS ${INSTRUMENT_COMMON_FLAGS} -dc -std=c++11 -Xptxas -cloning=no
)

cuda_compile(INJECT_O
    ${INJECT_SRC}
    OPTIONS ${INSTRUMENT_COMMON_FLAGS} -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions
)

set(CMAKE_CUDA_FLAGS "-arch=sm_${GPU_GEN_CODE}")
set(INSTRUMENT_LIB instrument_sm${GPU_GEN_CODE} PARENT_SCOPE)
set(INSTRUMENT_LIB instrument_sm${GPU_GEN_CODE})

add_library(${INSTRUMENT_LIB} SHARED ${INJECT_O} ${INSTRUMENT_O})

## see https://gist.github.com/gavinb/c993f71cf33d2354515c4452a3f8ef30
set_target_properties(${INSTRUMENT_LIB}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${INSTRUMENT_LIB}
    PRIVATE ${NVBIT_LIB} cuda cudart_static
)

if(WITH_IDEMPOTENCE_CHECK)
    add_subdirectory(idempotence-check)
endif()

if(WITH_SASS_DUMPER)
    add_subdirectory(sass-dump)
endif()

install(TARGETS ${INSTRUMENT_LIB} LIBRARY DESTINATION lib)
