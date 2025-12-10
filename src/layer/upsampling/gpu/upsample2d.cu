#include "../upsample2d.h"
#include <constants.h>

UpSample2DGPU::UpSample2DGPU(std::shared_ptr<Layer> prev)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&grad_input), in_x * in_y * in_z * sizeof(float));
}

std::tuple<int, int, int> UpSample2DGPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {x * SCALE_FACTOR, y * SCALE_FACTOR, z};
}

template <int scale>
__global__ void upsample2d_forward_kernel(float* output, const float* input, int out_row, int out_col) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= out_col || y >= out_row) return;

    int in_row = out_row / scale;
    int in_col = out_col / scale;
    output[c * out_row * out_col + y * out_col + x] = input[c * in_row * in_col + (y / scale) * in_col + (x / scale)];
}

void UpSample2DGPU::forward()
{
    m_prev->forward();
    auto [x, y, z] = dimension();

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize((x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                  (y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                   z);

    upsample2d_forward_kernel<SCALE_FACTOR><<<gridSize, blockSize>>>
                                    (m_output, m_prev->output(), y, x);

    CHECK(cudaGetLastError());
}

template <int scale>
__global__ void upsample2d_backward_kernel(float* grad_input, const float* grad_output, int in_row, int in_col) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= in_col || y >= in_row) return;

    int out_row = in_row * scale;
    int out_col = in_col * scale;

    float sum = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < scale; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < scale; ++dx)
            sum += grad_output[c * out_row * out_col + (y * scale + dy) * out_col + (x * scale + dx)];
    }
    grad_input[c * in_row * in_col + y * in_col + x] = sum;
}

void UpSample2DGPU::backward(float learning_rate, const float* grad_output)
{
    auto [x, y, z] = dimension();
    auto [in_x, in_y, _] = m_prev->dimension();

    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 gridSize((in_x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                  (in_y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                   z);

    upsample2d_backward_kernel<SCALE_FACTOR><<<gridSize, blockSize>>>
                                    (grad_input, grad_output, in_y, in_x);

    CHECK(cudaGetLastError());

    m_prev->backward(learning_rate, grad_input);
}
