#include <assert.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <condition_variable>
#include <thread>

/* nvbit interface file */
#include "nvbit.h"



struct Task {
    CUfunction f;
    CUcontext ctx;
    bool exit;
};

std::mutex mtx;
std::condition_variable cv;
std::queue<Task> task_queue;


void dump_sass_if_needed(CUcontext ctx, CUfunction f) {
    std::unique_lock<std::mutex> lock(mtx);
    Task t = {0};
    t.f = f;
    t.ctx = ctx;
    t.exit = false;
    task_queue.push(t);
    cv.notify_one();
}

void background_sass_dump_worker(std::string file_path) {
    std::unordered_set<CUfunction> already_saved;
    std::ofstream outStream(file_path);
    if (!outStream.is_open()) {
        printf("Cannot open file %s for sass dump.\n", file_path.c_str());
        exit(-1);
    }
    while (true) {
        Task t;
        {
            std::unique_lock<std::mutex> lock(mtx);
            while (task_queue.empty()) {
                cv.wait(lock);
            }
            t = task_queue.front();
            task_queue.pop();
        }
        if (t.exit) {
            break;
        }
        if (already_saved.find(t.f) != already_saved.end()) continue;
        already_saved.insert(t.f);
        /* Get related functions of the kernel (device function that can be called by the kernel) */
        std::vector<CUfunction> related_functions = nvbit_get_related_functions(t.ctx, t.f);
        /* add kernel itself to the related function vector */
        related_functions.push_back(t.f);
        for (auto f : related_functions) {
            const std::vector<Instr*>& instrs = nvbit_get_instrs(t.ctx, f);
            std::string func_name(nvbit_get_func_name(t.ctx, f));
            std::string mangled_name(nvbit_get_func_name(t.ctx, f, true));
            outStream << "Function : " << mangled_name << std::endl;
            outStream << "name : " << func_name << std::endl;
            size_t num_instrs = instrs.size();
            int w = 0;
            while (num_instrs != 0) {
                w += 1;
                num_instrs /= 16;
            }
            if (w <= 3) w = 3;
            for (auto instr : instrs) {
                outStream << "       /*" << std::setfill('0') << std::setw(w+1) << std::hex
                << (uint64_t)instr->getOffset() << "*/                  " << instr->getSass() << std::endl;
            }
            outStream << "                ..........             " << std::endl;
        }
    }

    // outStream.close();
}

/* This call-back is triggered every time a CUDA driver call is encountered.
* Here we can look for a particular CUDA driver call by checking at the
* call back ids  which are defined in tools_cuda_api_meta.h.
* This call back is triggered bith at entry and at exit of each CUDA driver
* call, is_exit=0 is entry, is_exit=1 is exit.
* */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                        const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        /* cast params to cuLaunch_params since if we are here we know these are
        * the right parameters type */
        cuLaunch_params *p = (cuLaunch_params *)params;
        if (is_exit) {
            return;
        }
        dump_sass_if_needed(ctx, p->f);
    }
}

std::thread* worker;

void nvbit_at_init() {
    
    std::string sass_file_path;
    GET_VAR_STR(sass_file_path, "SASS_FILE_PATH", 
        "The file path for saving the dumpped sass"
    );
    worker = new std::thread([&]() {
        background_sass_dump_worker(sass_file_path);
    });
}

void nvbit_at_term() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        Task t = {0};
        t.exit = true;
        task_queue.push(t);
        cv.notify_one();
    }
    worker->join();
    delete worker;
}
