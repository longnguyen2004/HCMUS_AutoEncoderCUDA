#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CHECK(error) { \
    const cudaError_t err = (error); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


inline double checkDiff(const float* cpu, const float* gpu, int n) {
    double sum_error = 0.0;
    
    for (int i = 0; i < n; i++)
        sum_error += std::abs(cpu[i] - gpu[i]);
    
    return sum_error / n;
}

class GPUTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};