find_package(CUDA REQUIRED)

set(CHECK_INCLUDE_DIR
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(CHECK_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/nvbit.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/kernel.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/checker.cu
)

set(CHECK_INJECT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/inject.cu
)

set(INSTRUMENT_COMMON_FLAGS
    ${INSTRUMENT_COMMON_FLAGS}
    -I ${CHECK_INCLUDE_DIR}
)

cuda_compile(CHECK_O
    ${CHECK_SRC}
    OPTIONS ${INSTRUMENT_COMMON_FLAGS} -dc -std=c++11 -Xptxas -cloning=no
)

cuda_compile(CHECK_INJECT_O
    ${CHECK_INJECT_SRC}
    OPTIONS ${INSTRUMENT_COMMON_FLAGS} -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions
)

add_library(check_instrument_sm${GPU_GEN_CODE} SHARED ${CHECK_O} ${CHECK_INJECT_O})

set_target_properties(check_instrument_sm${GPU_GEN_CODE}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(check_instrument_sm${GPU_GEN_CODE}
    PRIVATE ${NVBIT_LIB} cuda cudart_static
)

install(TARGETS check_instrument_sm${GPU_GEN_CODE} LIBRARY DESTINATION lib/idempotence_check)


# add cuda hook
set(HOOK_SRC
    ${CMAKE_CURRENT_LIST_DIR}/src/cuda_hook.cpp
)

add_library(idempotence_check SHARED ${HOOK_SRC})

target_include_directories(idempotence_check
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(idempotence_check
    PRIVATE dl check_instrument_sm${GPU_GEN_CODE}
)

target_compile_options(idempotence_check
    PRIVATE
    -fPIC
    -fvisibility=hidden
    
    -Wall
    # -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

target_link_options(idempotence_check
    PRIVATE -s -Wl,--exclude-libs,ALL
)

install(TARGETS idempotence_check LIBRARY DESTINATION lib/idempotence_check)
install(CODE "
    execute_process(
        COMMAND ln -s libidempotence_check.so libcuda.so.1
        COMMAND ln -s libcuda.so.1 libcuda.so
        WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib/idempotence_check
    )
    message(\"-- Installing: softlink created: libcuda.so.1 libcuda.so\")
")
