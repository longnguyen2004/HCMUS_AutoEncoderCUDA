#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void inline CHECK(const cudaError_t error) {
    if (error == cudaSuccess) return;
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
    exit(EXIT_FAILURE); 
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