# 用户文档

本文档介绍如何使用本工具实现 GPU (CUDA) 任务抢占式调度。


## API

REEF 提供以下 API

* [reef::enablePreemption(CUstream stream, RFconfig config)](#reefenablepreemptioncustream-stream-rfconfig-config)
* [reef::disablePreemption(CUstream stream)](#reefdisablepreemptioncustream-stream)
* [reef::preempt(CUstream stream, bool sync)](#reefpreemptcustream-stream-bool-sync)
* [reef::restore(CUstream stream)](#reefrestorecustream-stream)

### reef::enablePreemption(CUstream stream, RFconfig config)

为一个 cuda stream 开启抢占功能。**只有在调用 reef::enablePreemption 后，才可以调用 reef::preempt 以及 reef::restore** 。当对一个 cuda stream 调用 enablePreemption 后，REEF 会劫持这个 stream 中所有的异步 cuda 调用（例如，cuLaunchKernel, cuMemcpyAsync, cuMemsetAsync 等），并创建一个 *后台线程* 用于控制劫持的异步 cuda 调用。

参数：
* stream: 开启抢占功能的 cuda stream;
* RFconfig: 抢占功能的配置参数，默认使用 reef::defaultConfig。

RFconfig 定义：
```c
struct RFconfig
{
    size_t queueSize;       // vDeviceQueue (vDQ) 的队列大小；queueSize 越大，抢占的时延越高，应用执行的开销越低。
    size_t batchSize;       // 向 vDQ 提交指令的 batch 大小；batchSize 越大，应用执行的开销越低；batchSize 应该不超过 queueSize；
    int32_t taskTimeout;    // 设置 vDeviceQueue 的超时阈值；当一段时间没有任何指令被 vDeviceQueue 消费，则认为当前 stream 的任务已经完成，会执行相应回调函数通知调度器。Measured in microseconds (us). taskTimeout < 0 means disable timeout mechanism.
    PreemptLevel preemptLevel; // 支持的抢占机制等级。抢占机制等级越低，应用执行开销越低，抢占时延越高。
};


enum PreemptLevel
{
    PreemptHostQueue    = 0, // PreemptHostQueue 等级在任务抢占时，会等待 vDeviceQueue 中所有指令完成。拥有最低的应用执行开销、相对最高的抢占时延。
    PreemptDeviceQueue  = 1, // PreemptDeviceQueue 等级会对执行的 GPU kernel 进行插桩，在抢占时仅等待正在执行的 GPU kernel 完成正常执行，在 vDeviceQueue 中的其他 GPU kernel 会通过插桩代码退出执行。相比 PreemptHostQueue 具有更高的应用执行开销、更低的抢占时延。
    PreemptAll          = 2, // 当前版本不支持此等级。
};
```

### reef::disablePreemption(CUstream stream)

为一个 cuda stream 关闭抢占功能。**只有在调用 reef::enablePreemption 后，才可以调用 reef::disablePreemption**。reef::disablePreemption 会回收 REEF 为当前 stream 创建的所有资源(vHQ, 后台线程等）。

参数：
* stream: 关闭抢占功能的 cuda stream。

### reef::preempt(CUstream stream, bool sync)

抢占(暂停)一个 cuda stream 的执行。**只有在调用 reef::enablePreemption 后，才可以调用 reef::preempt**。 reef::preempt 会暂停当前 stream 中提交的异步命令的执行，但不会影响 cuda stream 的串行语义。当一个 cuda stream 被抢占后，后续向这个 stream 提交的异步指令都要等 stream 被恢复后才会执行。根据 reef::enablePreemption 设置的 PreemptLevel 的不同，reef::preempt 会执行不同的操作，例如在 PreemptHostQueue 等级下，reef::preempt 仅会暂停后台线程，而 PreemptDeviceQueue 等级下，reef::preempt 还会设置抢占flag。

参数：
* stream: 抢占的 cuda stream （必须已经调用过 reef::enablePreemption);
* sync: 是否等待抢占结束。当 sync 设置为 true 时，reef::preempt 会阻塞调用现场直到抢占结束（同步抢占），可用于测试抢占时延。当 sync 设置为 false 时，reef::preempt 会在后台完成抢占，不阻塞当前线程(异步抢占)。

### reef::restore(CUstream stream)

恢复一个被抢占的 cuda stream 的执行。**只有在调用 reef::preempt 后才可以调用 reef::restore**。 根据 reef::enablePreemption 设置的 PreemptLevel 的不同，reef::restore 会执行不同的操作，例如在 PreemptHostQueue 等级下，reef::restore 仅会恢复后台线程，而 PreemptDeviceQueue 等级下，reef::restore 还会重置抢占flag。

参数：
* stream: 恢复执行的 cuda stream （必须已经调用过 reef::preempt)。


## Example


REEF 的常见用法：

```c
// 1. 创建 cuda stream 
cudaStream_t lp_stream, hp_stream;
cudaCreateStream(&lp_stream); // 低优先级 stream
cudaCreateStream(&hp_stream); // 高优先级 stream

// 2. 为 stream 开启抢占机制
reef::RFconfig config {
    .queueSize = 32, 
    .batchSize = 8,
    .taskTimeout = -1,
    .preemptLevel = reef::PreemptHostQueue
};

reef::enablePreemption(lp_stream, config); 

// 3. 提交异步 CUDA 指令到低优先 stream 中
cudaMemcpyAsync(..., lp_stream); // 提交异步的 memcpy
kernel_lp<<<..., lp_stream>>>(...); // 提交异步的 kernel

// 4. 在异步 CUDA 指令执行的过程中，抢占正在执行的低优先级 stream
reef::preempt(hp_stream);

// 5. 向高优先级 stream 提交 CUDA 指令
// 在 lp_stream 被抢占后，hp_stream 的执行性能不会受到 lp_stream 干扰
kernel_hp<<<..., hp_stream>>>(...);
cudaStreamSynchronize(hp_stream);

// 6. 恢复低优先级 stream 的执行
reef::restore(lp_stream);
cudaStreamSynchronize(lp_stream);

// 7. 在程序结束时释放资源（可选）
reef::disablePreemption(lp_stream);
reef::disablePreemption(hp_stream);

```