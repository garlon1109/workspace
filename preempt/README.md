# TensorRT-Preempt



## Supported Platforms

- Host CPU:

| Architecture | Tested |
| ------------ | ------ |
| x86_64       | ✅      |
| aarch64      | ❌      |

- GPU:

| SM Compute Capability | Typical GPU | Tested |
| --------------------- | ----------- | ------ |
| 8.6                   | RTX3080     | ✅      |
| 7.0                   | V100        | ✅      |

- OS: Linux64



## Dependencies

- CUDA >= 11.3, 11.3.1 is recommended
- (Optional) TensorRT == 7.2.3.4, needed if you want to build TensorRT testcases
- GCC >= 5.3.0
- CMake >= 3.15
- (Optional) rustc >= 1.65.0, needed if you want to build scheduler

Docker environment is recommended (`nvidia-docker` should be installed):

```shell
# build docker image
docker build -t reef-trt7-rust docker/

# run
docker run -it --name reef-trt7-rust --gpus all -v ${PWD}:/workspace/tensorrt-preempt reef-trt7-rust /bin/bash
```

If you do not need scheduler, NVIDIA official docker image is recommended:

```shell
# TensorRT 7.2.3.4, CUDA 11.3.0, cuBLAS 11.5.1.101, cuDNN 8.2.0.51
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-05.html#rel_21-05
docker pull nvcr.io/nvidia/tensorrt:21.05-py3
docker run -it --name reef-trt7 --gpus all -v ${PWD}:/workspace/tensorrt-preempt nvcr.io/nvidia/tensorrt:21.05-py3 /bin/bash
```



## Quick Build & Install

```shell
# BUILD_TYPE	= Debug / Release
# VERBOSE		= ON / OFF : enable verbose makefile
# HOOK_LOG		= ON / OFF : enable hook logs
# REEF_LOG		= ON / OFF : enable reef logs
# WITH_TEST		= ON / OFF : whether to build cudareef tests
# WITH_SCHED	= ON / OFF : whether to build scheduler
# WITH_IDEMPOTENCE_CHECK    = ON / OFF : build with idempotence checker
# Please refer to Makefile for default values.
make build BUILD_TYPE=Release WITH_TEST=ON WITH_SCHED=ON
```

If build successfully, REEF will be installed to `output` folder. Instrument tools will be built into `output/lib/libinstrument_smxx.so`, REEF will be built into `output/lib/libcudareef.so`, `output/lib/libcuda.so` and `output/lib/libcuda.so.1` are softlinks of `output/lib/libcudareef.so`.



## Build Options

You can use CMake to build the library.

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHOOK_LOG=OFF -DREEF_LOG=OFF \
    -DWITH_TEST=ON -DWITH_SCHED=ON -DWITH_IDEMPOTENCE_CHECK=ON -DCMAKE_INSTALL_PREFIX=output
cmake --build --target install -- -j4
```

Binary-level instrument needs to be optimized to reduce execution overhead.
If your GPU architecture is not supported (or reef has not been optimized for), you can still build and use reef. However, default preempt level will be set to `PreemptHostQueue` to reduce overhead, this will disable instrument and preemption on device queue. If you still want to use `PreemptDeviceQueue` or higher preempt level, default instrument method without optimization will be applied. e.g.:

If you do not want to use the scheduler, or you just do no want to install rust, set `WITH_SCHED` to `OFF`. e.g.:
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHOOK_LOG=OFF -DREEF_LOG=OFF \
    -DWITH_TEST=ON -DWITH_SCHED=OFF -DWITH_IDEMPOTENCE_CHECK=ON -DCMAKE_INSTALL_PREFIX=output
```



## Usage

To call APIs provided by REEF, `output/include/reef.h` should be included, `output/lib/libcudareef.so` and `output/lib/libinstrument_smxx.so` should be linked with your executables. You can have a look at `test/tensorrt` as an example.

### APIs provided by REEF

```c++
//!
//! \brief enable preemption capability on a cuda stream,
//! should be call before calling preempt() and restore()
//!
//! \param cuStream The cuStream that should enable preemption capability.
//! \param config The configuration that controls how REEF acts.
//!        Default config will be used if omitted.
//!
//! \return REEF_SUCCESS if succeeded, others if failed
//!
RFresult enablePreemption(CUstream cuStream, RFconfig config=defaultConfig);

//!
//! \brief diable preemption capability on a cuda stream
//!
//! \param cuStream The cuStream that have already enabled preemption capability.
//!
//! \return REEF_SUCCESS if succeeded, others if failed
//!
RFresult disablePreemption(CUstream cuStream);

//!
//! \brief preempt a running cuda stream, all submitted tasks on this stream will be suspended.
//!
//! \param cuStream The cuStream that have already enabled preemption capability.
//! \param synchronize Whether to wait until the stream is stopped and all tasks are suspended.
//!
//! \return REEF_SUCCESS if succeeded, others if failed
//!
RFresult preempt(CUstream cuStream, bool synchronize=false);

//!
//! \brief restore a preempted cuda stream, all submitted tasks on this stream will be resumed.
//!
//! \param cuStream The cuStream that have already enabled preemption capability.
//!
//! \return REEF_SUCCESS if succeeded, others if failed
//!
RFresult restore(CUstream cuStream);
```

### Run a REEF executable

Many CUDA executables will dynamically load `libcuda.so` or `libcuda.so.1` under `/usr/lib/x86_64-linux-gnu`, others (for example, TensorRT applications) will use `dlopen()` to load them. REEF intercepts CUDA calls by exposing all CUDA symbols (or APIs) before the real `libcuda.so` in `LD_LIBRARY_PATH`.

```shell
# add output/lib (or other places you placed REEF libs) to env
# be aware the libcudareef.so and its softlinks should be prior to the real libcuda.so
export LD_LIBRARY_PATH=${PWD}/output/lib:$LD_LIBRARY_PATH

# run your executable
```

You may refer to `Makefile` as an example.

### Example

See `example/example.cu`.

To run the example:
```bash
# make sure you have build libcudareef.so
cd example
make
make run
```