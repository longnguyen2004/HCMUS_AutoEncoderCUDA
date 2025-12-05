#include "../maxpool2d.h"
#include <mdspan/mdspan.hpp>
#include <constants.h>

MaxPool2DGPU::MaxPool2DGPU(std::shared_ptr<Layer> prev)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    CHECK(cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), in_x * in_y * in_z * sizeof(float)));
}

std::tuple<int, int, int> MaxPool2DGPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {
        (x + MAXPOOL2D_STRIDE - 1) / MAXPOOL2D_STRIDE,
        (y + MAXPOOL2D_STRIDE - 1) / MAXPOOL2D_STRIDE,
        z
    };
}

template <int stride>
__global__ void maxpool2d_forward_kernel(float* output, const float* input, size_t input_col, size_t input_row, 
                                            size_t input_size, size_t output_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    int output_col = (input_col + stride - 1) / stride;
    int output_row = (input_row + stride - 1) / stride;
    
    if (x >= output_col || y >= output_row)
        return;

    int x_input = stride * x;
    int y_input = stride * y;
    const float* cur_input = input + (c * input_size);
    float* cur_output = output + (c * output_size);

    float max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < stride; ++i) {
        #pragma unroll
        for (int j = 0; j < stride; ++j) {
            int cur_y = y_input + i;
            int cur_x = x_input + j;
            if (cur_y < input_row && cur_x < input_col)
                max = fmaxf(max, cur_input[cur_y * input_col + cur_x]);
        }
    }
    cur_output[y * output_col + x] = max;
}

void MaxPool2DGPU::forward() {
    m_prev->forward();
    auto [in_x, in_y, z] = m_prev->dimension();
    auto [out_x, out_y, _] = dimension();
    
    int input_size = in_x * in_y;
    int output_size = out_x * out_y;

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize(
            (out_x + blockSize.x - 1) / blockSize.x,
            (out_y + blockSize.y - 1) / blockSize.y,
            z
        );

    maxpool2d_forward_kernel<MAXPOOL2D_STRIDE><<<gridSize, blockSize>>>(
            m_output,
            m_prev->output(),
            in_x,
            in_y,
            input_size,
            output_size
        );

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

template <int stride>
__global__ void maxpool2d_backward_kernel(const float* grad_output, float* grad_input, const float* input, const float* output,
    int input_row, int input_col, int input_size, int output_size) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    int output_col = (input_col + stride - 1) / stride;
    int output_row = (input_row + stride - 1) / stride;
    
    if (x >= output_col || y >= output_row) return;
    
    int x_input = stride * x;
    int y_input = stride * y;
    
    const float* cur_input = input + (c * input_size);
    const float* cur_output = output + (c * output_size);
    grad_input  += c * input_size;
    grad_output += c * output_size;
    
    #pragma unroll
    for (int i = 0; i < stride; ++i) {
        #pragma unroll
        for (int j = 0; j < stride; ++j) {
            int cur_y = y_input + i;
            int cur_x = x_input + j;
            if (cur_y < input_row && cur_x < input_col) {
                if (cur_input[cur_y * input_col + cur_x] == cur_output[y * output_col + x])
                    grad_input[cur_y * input_col + cur_x] = grad_output[y * output_col + x];
            }
        }
    }
}

void MaxPool2DGPU::backward(float learning_rate, const float* grad_output) {
    auto [in_x, in_y, z] = m_prev->dimension();
    auto [out_x, out_y, _] = dimension();

    int input_size = in_x * in_y;
    int output_size = out_x * out_y;

    CHECK(cudaMemset(grad_input, 0, in_x * in_y * z * sizeof(float)));

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize(
        (out_x + blockSize.x - 1 ) / blockSize.x,
        (out_y + blockSize.y - 1) / blockSize.y,
        z
    );

    maxpool2d_backward_kernel<MAXPOOL2D_STRIDE><<<gridSize, blockSize>>>(
        grad_output,
        grad_input,
        m_prev->output(),
        m_output,
        in_y, in_x, input_size, output_size);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());    
    m_prev->backward(learning_rate, grad_input);
}
