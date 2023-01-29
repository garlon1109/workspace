#pragma once

#include <ctime>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <functional>

#include "model.h"
#include "reef/reef.h"

#define LATENCY_TEST_TIME       1000
#define CORRECTNESS_TEST_TIME   500

int64_t timeElapsedNs(std::function<void ()> call)
{
    auto start = std::chrono::system_clock::now();
    call();
    auto stop = std::chrono::system_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return ns;
}

void warmUp(TRTModel &model, cudaStream_t stream)
{
    for (int i = 0; i < LATENCY_TEST_TIME; ++i) model.inferSync(stream);
}

int64_t inferLatencyNs(TRTModel &model, cudaStream_t stream, int64_t* enqueue = nullptr)
{
    int64_t total_enqueue_time = 0;
    int64_t exec_time = timeElapsedNs([&]() -> void {
        for (int i = 0; i < LATENCY_TEST_TIME; ++i) {
            int64_t enqueue_time = timeElapsedNs([&]() {
                model.inferAsync(stream);
            });
            total_enqueue_time += enqueue_time;
            cudaStreamSynchronize(stream);
        }
    }) / LATENCY_TEST_TIME;
    if (enqueue != nullptr) *enqueue = total_enqueue_time / LATENCY_TEST_TIME;
    return exec_time;
}

void preemptTest(TRTModel &reefModel, cudaStream_t reefStream)
{
    srand(time(0));
    int64_t preemptNs = 0;
    int64_t restoreNs = 0;
    std::atomic_bool running;
    running.store(true);
    std::thread thread([&]() -> void {
        while (running.load()) reefModel.inferSync(reefStream);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int i = 0; i < LATENCY_TEST_TIME; ++i) {
        std::this_thread::sleep_for(std::chrono::microseconds(rand() % 1000));
        preemptNs += timeElapsedNs([&]() -> void { reef::preempt(reefStream, true); });
        std::this_thread::sleep_for(std::chrono::microseconds(rand() % 1000));
        restoreNs += timeElapsedNs([&]() -> void { reef::restore(reefStream); });
    }

    running.store(false);
    thread.join();

    printf("[RESULT] [TEST] preempt time: %.2fus\n", float(preemptNs / LATENCY_TEST_TIME) / 1000);
    printf("[RESULT] [TEST] restore time: %.2fus\n", float(restoreNs / LATENCY_TEST_TIME) / 1000);
}

bool checkOutput(TRTModel &reefModel, int testcase)
{
    bool pass = true;
    for (auto outputTensor : reefModel.outputTensors()) {
        if (!outputTensor.second->checkCorrect()) {
            pass = false;
            printf("[RESULT] [TEST] [FAIL] testcase %d FAIL: output tensor %s does not match\n", testcase, outputTensor.first.c_str());
        }
    }
    return pass;
}

void correctnessTest(TRTModel &reefModel, cudaStream_t reefStream)
{
    printf("[INFO] [TEST] test size: %d\n", CORRECTNESS_TEST_TIME);
    srand(time(0));
    int64_t enqueueNs = 0;
    int64_t executeNs = 0;
    {
        int64_t enqueueNsTotal = 0;
        int64_t executeNsTotal = 0;
        const int testCnt = 10;

        for (int i = 0; i < testCnt; ++i) {
            auto start = std::chrono::system_clock::now();
            reefModel.inferAsync(reefStream);
            auto enqueueEnd = std::chrono::system_clock::now();
            ASSERT_CUDA_ERROR(cudaStreamSynchronize(reefStream));
            auto executeEnd = std::chrono::system_clock::now();

            enqueueNsTotal += std::chrono::duration_cast<std::chrono::nanoseconds>(enqueueEnd - start).count();
            executeNsTotal += std::chrono::duration_cast<std::chrono::nanoseconds>(executeEnd - start).count();
        }

        enqueueNs = enqueueNsTotal / testCnt;
        executeNs = executeNsTotal / testCnt;
        printf("[INFO] [TEST] enqueue time: %ldus, execute time: %ldus\n", enqueueNs / 1000, executeNs / 1000);
    }
    {
        // TEST CASE 1
        // preempt
        //    |  restore
        //    |     |
        //    v     v
        // -----------|<------ enqueue + execute ------>|<------------------ execute ------------------>|-----------
        reef::preempt((CUstream)reefStream, true);
        reef::restore((CUstream)reefStream);
        reefModel.inferSync(reefStream);
        
        if (checkOutput(reefModel, 1)) {
            printf("[RESULT] [TEST] [PASS] testcase 1 PASSED\n");
        }
    }
    {
        // TEST CASE 2
        // preempt                 restore
        //    |                       |
        //    v                       v
        // -----------|<-- enqueue -->|<-- enqueue + execute -->|<------------------ execute ------------------>|-----------
        bool pass = true;
        for (int i = 0; i < CORRECTNESS_TEST_TIME; ++i) {
            reef::preempt((CUstream)reefStream, true);
            int64_t sleepNs = rand() % enqueueNs;
            auto start = std::chrono::system_clock::now();
            std::thread thread([&]() -> void {
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(sleepNs));
                reef::restore((CUstream)reefStream);
            });
            reefModel.inferSync(reefStream);
            thread.join();

            if (!checkOutput(reefModel, 2)) {
                pass = false;
                printf("[INFO] [TEST] [FAIL] restore after preempt time: %ldus\n", sleepNs / 1000);
                break;
            }
        }
        if (pass) {
            printf("[RESULT] [TEST] [PASS] testcase 2 PASSED\n");
        }
    }
    {
        // TEST CASE 3
        // preempt                                      restore
        //    |                                            |
        //    v                                            v
        // -----------|<------ enqueue ------>|<-- idle -->|<------------- execute ------------->|-----------
        bool pass = true;
        for (int i = 0; i < CORRECTNESS_TEST_TIME; ++i) {
            reef::preempt((CUstream)reefStream, true);
            int64_t sleepNs = (rand() % (executeNs - enqueueNs)) + enqueueNs;
            auto start = std::chrono::system_clock::now();
            std::thread thread([&]() -> void {
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(sleepNs));
                reef::restore((CUstream)reefStream);
            });
            reefModel.inferSync(reefStream);
            thread.join();

            if (!checkOutput(reefModel, 3)) {
                pass = false;
                printf("[INFO] [TEST] [FAIL] restore after preempt time: %ldus\n", sleepNs / 1000);
                break;
            }
        }
        if (pass) {
            printf("[RESULT] [TEST] [PASS] testcase 3 PASSED\n");
        }
    }
    {
        // TEST CASE 4
        //                                   preempt         restore
        //                                      |               |
        //                                      v               v
        // -----------|<-- enqueue + execute -->|<-- enqueue -->|<-- enqueue + execute -->|<----------- execute ----------->|-----------
        bool pass = true;
        for (int i = 0; i < CORRECTNESS_TEST_TIME; ++i) {
            int64_t preemptSleepNs = rand() % enqueueNs;
            int64_t restoreSleepNs = (rand() % (enqueueNs - preemptSleepNs)) + preemptSleepNs;
            auto start = std::chrono::system_clock::now();
            std::thread thread([&]() -> void {
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(preemptSleepNs));
                reef::preempt((CUstream)reefStream, false);
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(restoreSleepNs));
                reef::restore((CUstream)reefStream);
            });
            reefModel.inferSync(reefStream);
            thread.join();

            if (!checkOutput(reefModel, 4)) {
                pass = false;
                printf("[INFO] [TEST] [FAIL] preempt after enqueue time: %ldus, restore after enqueue time: %ldus\n", preemptSleepNs / 1000, restoreSleepNs / 1000);
                break;
            }
        }
        if (pass) {
            printf("[RESULT] [TEST] [PASS] testcase 4 PASSED\n");
        }
    }
    {
        // TEST CASE 5
        //                                   preempt                      restore
        //                                      |                            |
        //                                      v                            v
        // -----------|<-- enqueue + execute -->|<-- enqueue -->|<-- idle -->|<------------------ execute ------------------>|-----------
        bool pass = true;
        for (int i = 0; i < CORRECTNESS_TEST_TIME; ++i) {
            int64_t preemptSleepNs = rand() % enqueueNs;
            int64_t restoreSleepNs = (rand() % executeNs) + enqueueNs;
            auto start = std::chrono::system_clock::now();
            std::thread thread([&]() -> void {
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(preemptSleepNs));
                reef::preempt((CUstream)reefStream, false);
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(restoreSleepNs));
                reef::restore((CUstream)reefStream);
            });
            reefModel.inferSync(reefStream);
            thread.join();

            if (!checkOutput(reefModel, 5)) {
                pass = false;
                printf("[INFO] [TEST] [FAIL] preempt after enqueue time: %ldus, restore after enqueue time: %ldus\n", preemptSleepNs / 1000, restoreSleepNs / 1000);
                break;
            }
        }
        if (pass) {
            printf("[RESULT] [TEST] [PASS] testcase 5 PASSED\n");
        }
    }
    {
        // TEST CASE 6
        //                                                   preempt      restore
        //                                                      |            |
        //                                                      v            v
        // -----------|<-- enqueue + execute -->|<-- execute -->|<-- idle -->|<------------------ execute ------------------>|-----------
        bool pass = true;
        for (int i = 0; i < CORRECTNESS_TEST_TIME; ++i) {
            int64_t preemptSleepNs = (rand() % (executeNs - enqueueNs)) + enqueueNs;
            int64_t restoreSleepNs = (rand() % executeNs) + preemptSleepNs;
            auto start = std::chrono::system_clock::now();
            std::thread thread([&]() -> void {
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(preemptSleepNs));
                reef::preempt((CUstream)reefStream, false);
                std::this_thread::sleep_until(start + std::chrono::nanoseconds(restoreSleepNs));
                reef::restore((CUstream)reefStream);
            });
            reefModel.inferSync(reefStream);
            thread.join();

            if (!checkOutput(reefModel, 6)) {
                pass = false;
                printf("[INFO] [TEST] [FAIL] preempt after enqueue time: %ldus, restore after enqueue time: %ldus", preemptSleepNs / 1000, restoreSleepNs / 1000);
                break;
            }
        }
        if (pass) {
            printf("[RESULT] [TEST] [PASS] testcase 6 PASSED\n");
        }
    }
}