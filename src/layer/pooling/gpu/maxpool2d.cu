#include "../maxpool2d.h"
#include <helper/gpu_helper.h>
#include <mdspan/mdspan.hpp>
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

std::tuple<int, int, int> MaxPool2DGPU::dimension() const
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
    
    if (x >= output_col || y >= output_row)
        return;

    int x_input = stride * x;
    int y_input = stride * y;
    
    float max = -INFINITY;
    for (int i = 0; i < stride; ++i) {
        for (int j = 0; j < stride; ++j) {
            int cur_y = y_input + i;
            int cur_x = x_input + j;
            if (cur_y < input_row && cur_x < input_col)
                max = fmaxf(max, input[cur_y * input_col + cur_x]);
        }
    }
    output[y * output_col + x] = max;
}

void MaxPool2DGPU::forward() {
    m_prev->forward();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [out_x, out_y, out_z] = dimension();
    size_t input_size = in_x * in_y;
    size_t output_size = out_x * out_y;

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize(
        (out_x + blockSize.x - 1 ) / blockSize.x,
        (out_y + blockSize.y - 1) / blockSize.y
    );

    for (int c = 0; c < in_z; ++c)
        maxpool2d_forward_kernel<<<gridSize, blockSize>>>(m_output + c * output_size, 
                                                          m_prev->output() + c * input_size, in_x, in_y, m_stride);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

__global__ void maxpool2d_backward_kernel(const float* grad_output, float* grad_input, const float* input, const float* output,
    size_t input_row, size_t input_col, int stride) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int output_col = (input_col + stride - 1) / stride;
    int output_row = (input_row + stride - 1) / stride;
    
    if (x >= output_col || y >= output_row) return;
    
    int x_input = stride * x;
    int y_input = stride * y;
    
    float max_val = output[y * output_col + x];
    float grad = grad_output[y * output_col + x];
    
    for (int i = 0; i < stride; ++i) {
        for (int j = 0; j < stride; ++j) {
            int cur_y = y_input + i;
            int cur_x = x_input + j;
            if (cur_y < input_row && cur_x < input_col) {
                if (input[cur_y * input_col + cur_x] == max_val)
                    grad_input[cur_y * input_col + cur_x] = grad;
            }
        }
    }
}

void MaxPool2DGPU::backward(float learning_rate, const float* grad_output) {
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [out_x, out_y, out_z] = dimension();
    float* grad_input;

    size_t input_size = in_x * in_y;
    size_t output_size = out_x * out_y;

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize(
        (out_x + blockSize.x - 1 ) / blockSize.x,
        (out_y + blockSize.y - 1) / blockSize.y
    );

    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), in_x * in_y * in_z * sizeof(float)));
    CHECK(cudaMemset(grad_input, 0, in_x * in_y * in_z * sizeof(float)));

    for (size_t c = 0; c < in_z; ++c)
        maxpool2d_backward_kernel<<<gridSize, blockSize>>>(
            grad_output + c * output_size,
            grad_input + c * input_size,
            m_prev->output() + c * input_size,
            m_output + c * output_size,
            in_y, in_x, m_stride);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
    CHECK(cudaFree(grad_input));
}
