set(SASS_DUMP_SOURCE
    ${CMAKE_CURRENT_LIST_DIR}/src/nvbit.cu
)

cuda_compile(SASS_DUMP_O
    ${SASS_DUMP_SOURCE}
    OPTIONS ${INSTRUMENT_COMMON_FLAGS} -dc -std=c++11 -Xptxas -cloning=no
)

add_library(sass_dumper SHARED ${SASS_DUMP_O})

target_include_directories(sass_dumper
    PRIVATE
    ${INSTRUMENT_INCLUDE_DIR}
)

set_target_properties(sass_dumper
PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
   CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(sass_dumper
PRIVATE ${NVBIT_LIB} cuda cudart_static pthread
)

install(TARGETS sass_dumper LIBRARY DESTINATION lib)
