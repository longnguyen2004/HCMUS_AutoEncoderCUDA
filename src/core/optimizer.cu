#include "optimizer.h"
#include <constants.h>
#include <helper/gpu_helper.h>
#include <cmath>
#include <vector>

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
    CHECK(cudaGetLastError());
}

// Clip gradients element-wise to [-clip_value, +clip_value]
__global__ void clipGradientsKernel(float* gradients, float clip_value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gradients[idx] = fmaxf(-clip_value, fminf(clip_value, gradients[idx]));
    }
}

void clipGradientsGPU(float* gradients, float clip_value, int n) {
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    clipGradientsKernel<<<gridSize, blockSize>>>(gradients, clip_value, n);
    CHECK(cudaGetLastError());
}

// Compute squared norm reduction kernel
__global__ void computeSquaredNormKernel(const float* gradients, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load and compute squared gradient
    float sum = 0.0f;
    if (idx < n) {
        float g = gradients[idx];
        sum = g * g;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Scale gradients by a factor
__global__ void scaleGradientsKernel(float* gradients, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gradients[idx] *= scale;
    }
}

void clipGradientsByNormGPU(float* gradients, float max_norm, int n) {
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    // Allocate memory for partial sums
    float* d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, gridSize.x * sizeof(float)));
    
    // Compute squared norm
    computeSquaredNormKernel<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(
        gradients, d_partial_sums, n);
    CHECK(cudaGetLastError());
    
    // Copy partial sums to host and compute final norm
    std::vector<float> h_partial_sums(gridSize.x);
    CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, 
                     gridSize.x * sizeof(float), cudaMemcpyDeviceToHost));
    
    float squared_norm = 0.0f;
    for (float val : h_partial_sums) {
        squared_norm += val;
    }
    float norm = std::sqrt(squared_norm);
    
    // Scale if norm exceeds max_norm
    if (norm > max_norm) {
        float scale = max_norm / norm;
        scaleGradientsKernel<<<gridSize, blockSize>>>(gradients, scale, n);
        CHECK(cudaGetLastError());
    }
    
    CHECK(cudaFree(d_partial_sums));
}
