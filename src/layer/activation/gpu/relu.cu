#include "../relu.h"

__global__ void relu_forward_kernel(float* out, const float* in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    float derivative = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * derivative;
}

ReluGPU::ReluGPU(std::shared_ptr<Layer> prev): m_prev(prev)
{
    auto [x, y, z] = m_prev->dimension();
    cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float));
}

ReluGPU::~ReluGPU()
{
    cudaFree(m_output);
}

const float* ReluGPU::output() const
{
    return m_output;
}

std::tuple<int, int, int> ReluGPU::dimension() const
{
    return m_prev->dimension();
}

size_t ReluGPU::paramCount() const
{
    return 0;
}

void ReluGPU::forward()
{
    m_prev->forward();
    auto [x, y, z] = m_prev->dimension();
    auto size = x * y * z;
    dim3 blockSize(32);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    relu_forward_kernel<<<gridSize, blockSize>>>(m_output, m_prev->output(), size);
    // TODO: sync
}

