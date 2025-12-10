#include "../relu.h"
#include <constants.h>
#include <helper/gpu_helper.h>

__global__ void relu_forward_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    float derivative = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * derivative;
}

ReluGPU::ReluGPU(std::shared_ptr<Layer> prev) 
{
    m_prev = prev;
    auto [x, y, z] = m_prev->dimension();
    size_t bytes = x * y * z * sizeof(float);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&m_output), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), bytes));
}

std::tuple<int, int, int> ReluGPU::dimension() const
{
    return m_prev->dimension();
}

void ReluGPU::forward()
{
    m_prev->forward();
    auto [x, y, z] = m_prev->dimension();
    auto size = x * y * z;
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    relu_forward_kernel<<<gridSize, blockSize>>>(m_output, m_prev->output(), size);
    CHECK(cudaGetLastError());
}

void ReluGPU::backward(float learning_rate, const float* grad_output) {
    auto [x, y, z] = m_prev->dimension();
    int size = x * y * z;

    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    relu_backward_kernel<<<gridSize, blockSize>>>(grad_output, m_prev->output(), grad_input, size);
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
}

