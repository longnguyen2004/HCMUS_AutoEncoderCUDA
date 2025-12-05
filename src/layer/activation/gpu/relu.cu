#include "../relu.h"
#include <constants.h>
#include <helper/gpu_helper.h>

__global__ void relu_forward_kernel(float* out, const float* in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    float derivative = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * derivative;
}

ReluGPU::ReluGPU(std::shared_ptr<Layer> prev): m_prev(prev)
{
    auto [x, y, z] = m_prev->dimension();
    CHECK(cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float)));
}

ReluGPU::~ReluGPU()
{
    CHECK(cudaFree(m_output));
}

const float* ReluGPU::output() const
{
    return m_output;
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
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

void ReluGPU::backward(float learning_rate, const float* grad_output) {
    auto [x, y, z] = m_prev->dimension();
    size_t size = x * y * z;
    float* grad_input;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), size * sizeof(float)));
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    relu_backward_kernel<<<gridSize, blockSize>>>(grad_output, m_prev->output(), grad_input, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
    CHECK(cudaFree(grad_input));
}

