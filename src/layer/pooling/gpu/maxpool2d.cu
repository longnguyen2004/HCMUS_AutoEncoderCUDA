#include "../maxpool2d.h"
#include <helper/gpu_helper.h>
#include <constants.h>

MaxPool2DGPU::MaxPool2DGPU(std::shared_ptr<Layer> prev, int stride)
    : m_prev(prev), m_stride(stride) 
{
    auto [x, y, z] = this->dimension();
    CHECK(cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float)));
}

MaxPool2DGPU::~MaxPool2DGPU() 
{
    cudaFree(m_output);
}

std::tuple<int, int, int> MaxPool2DCPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {
        (x + m_stride - 1) / m_stride,
        (y + m_stride - 1) / m_stride,
        z
    };
}

__global__ void maxpool2d_forward_kernel(float* output, const float* input, size_t input_col, size_t input_row, int stride) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int output_col = (input_col + stride - 1) / stride;
    int output_row = (input_row + stride - 1) / stride;
    int x_input = stride * x;
    int y_input = stride * y;
    if (x_input >= input_row || y_input >= input_col)
        return;
    float max = -INFINITY;
    for (int i = 0; i < stride; ++i) {
        for (int j = 0; j < stride; ++j)
            max = fmaxf(max, input[y_input * input_col + x_input]);
    }
    output[y * output_col + x] = max;
}

void MaxPool2DGPU::forward() {
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [out_x, out_y, out_z] = dimension();
    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize = (
        (out_x + blockSize.x - 1 ) / blockSize.x,
        (out_y + blockSize.y - 1) / blockSize.y
    );
    maxpool2d_forward_kernel<<<gridSize, blockSize>>>(m_output, m_prev->output(), in_x, in_y, m_stride);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

__global__ void maxpool2d_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    
}

void MaxPool2DGPU::backward(float learning_rate, const float* grad_output) {
    // TODO: calculate grad_input
    auto [x, y, z] = m_prev->dimension();
    size_t size = x * y * z;
    float* grad_input;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), size * sizeof(float)));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
}
