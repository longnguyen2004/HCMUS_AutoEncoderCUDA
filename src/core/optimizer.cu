#include "optimizer.h"
#include <constants.h>
#include <helper/gpu_helper.h>

__global__ void updateWeightsKernel(float* params, const float* gradients, float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= learning_rate * gradients[idx];
    }
}

void updateWeightsGPU(float* params, const float* gradients, float learning_rate, int n) {
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    updateWeightsKernel<<<gridSize, blockSize>>>(params, gradients, learning_rate, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}
