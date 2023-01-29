cmake_minimum_required(VERSION 3.15)

project(CUDAREEF LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

option(HOOK_LOG "whether to enable hook logs" OFF)
option(REEF_LOG "whether to enable reef logs" OFF)
option(WITH_TEST "whether to build cudareef tests" ON)
option(WITH_SCHED "whether to build scheduler" ON)
option(WITH_IDEMPOTENCE_CHECK "whether to build idempotence checker" ON)

set(CMAKE_C_FLAGS "-std=c11")
set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
set(CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

if(HOOK_LOG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHOOK_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHOOK_DEBUG")
endif()

if(REEF_LOG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DREEF_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DREEF_DEBUG")
endif()

if(WITH_IDEMPOTENCE_CHECK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIDEMPOTENCE_CHECK")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIDEMPOTENCE_CHECK")
endif()

# target: ${INSTRUMENT_LIB}
add_subdirectory(instrument)

if(WITH_SCHED)
    # target: preempt_client
    add_subdirectory(third_party/corrosion)
    corrosion_import_crate(MANIFEST_PATH scheduler/Cargo.toml)
    
    file(GLOB_RECURSE CUDAREEF_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
    )

    add_library(cudareef SHARED ${CUDAREEF_SRCS})

    target_include_directories(cudareef
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/scheduler/target
    )

    target_link_libraries(cudareef
        PRIVATE dl ${INSTRUMENT_LIB} preempt_client
    )
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNO_SCHED")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_SCHED")

    file(GLOB_RECURSE CUDAREEF_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/scheduler/dummy/*.c
        ${CMAKE_CURRENT_LIST_DIR}/scheduler/dummy/*.cpp
    )

    add_library(cudareef SHARED ${CUDAREEF_SRCS})

    target_include_directories(cudareef
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/scheduler/dummy
    )

    target_link_libraries(cudareef
        PRIVATE dl ${INSTRUMENT_LIB}
    )
endif()

target_compile_options(cudareef
    PRIVATE
    -fPIC
    
    -Wall
    # -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

target_link_options(cudareef
    PRIVATE -Wl,--exclude-libs,ALL
)

if(WITH_TEST)
    add_subdirectory(test)
endif()

install(TARGETS cudareef LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/reef/reef.h DESTINATION include)
install(CODE "
    execute_process(
        COMMAND ln -s libcudareef.so libcuda.so.1
        COMMAND ln -s libcuda.so.1 libcuda.so
        WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
    )
    message(\"-- Installing: softlink created: libcuda.so.1 libcuda.so\")
")
