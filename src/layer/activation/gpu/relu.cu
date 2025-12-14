#include "../relu.h"
#include <constants.h>
#include <helper/gpu_helper.h>

__global__ void relu_forward_kernel(float* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vec_size = size / 4;
    float4* out_vec = reinterpret_cast<float4*>(out);
    const float4* in_vec = reinterpret_cast<const float4*>(in);

    for (int i = idx; i < vec_size; i += stride) {
        float4 v = in_vec[i];
        v.x = fmaxf(0.0f, v.x);
        v.y = fmaxf(0.0f, v.y);
        v.z = fmaxf(0.0f, v.z);
        v.w = fmaxf(0.0f, v.w);
        out_vec[i] = v;
    }

    int tail_start = vec_size * 4;
    for (int i = tail_start + idx; i < size; i += stride) {
        out[i] = fmaxf(0.0f, in[i]);
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vec_size = size / 4;
    const float4* grad_out_vec = reinterpret_cast<const float4*>(grad_output);
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* grad_in_vec = reinterpret_cast<float4*>(grad_input);

    for (int i = idx; i < vec_size; i += stride) {
        float4 g = grad_out_vec[i];
        float4 v = in_vec[i];
        float4 r;
        
        r.x = (v.x > 0.0f) ? g.x : 0.0f;
        r.y = (v.y > 0.0f) ? g.y : 0.0f;
        r.z = (v.z > 0.0f) ? g.z : 0.0f;
        r.w = (v.w > 0.0f) ? g.w : 0.0f;
        
        grad_in_vec[i] = r;
    }

    int tail_start = vec_size * 4;
    for (int i = tail_start + idx; i < size; i += stride) {
        float derivative = (input[i] > 0.0f) ? 1.0f : 0.0f;
        grad_input[i] = grad_output[i] * derivative;
    }
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
    int size = x * y * z;
    
    int vec_size = size / 4;
    int num_blocks = (vec_size == 0) ? 1 : (vec_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize(num_blocks);
    relu_forward_kernel<<<gridSize, blockSize>>>(m_output, m_prev->output(), size);
    CHECK(cudaGetLastError());
}

void ReluGPU::backward(float learning_rate, const float* grad_output) {
    auto [x, y, z] = m_prev->dimension();
    int size = x * y * z;

    int vec_size = size / 4;
    int num_blocks = (vec_size == 0) ? 1 : (vec_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize(num_blocks);
    relu_backward_kernel<<<gridSize, blockSize>>>(grad_output, m_prev->output(), grad_input, size);
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
}