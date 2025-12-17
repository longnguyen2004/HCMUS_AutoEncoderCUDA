#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <core/optimizer.h>
#include <cuda_runtime.h>
#include <constants.h>
#include <helper/gpu_helper.h>

template <int kernel_width, bool flip_kernel>
__device__ __forceinline__
float im2col_map_coords(const float* __restrict__ src, const int c, const int y_in, const int x, const int row, const int col) {
    constexpr int pad = kernel_width / 2;
    constexpr int k_sq = kernel_width * kernel_width;
    const int y = flip_kernel ? (k_sq - 1 - y_in) : y_in;
    const int x_mapped = (x % col) - pad + (y % kernel_width);
    const int y_mapped = (x / col) - pad + (y / kernel_width);
    if ((unsigned)x_mapped >= (unsigned)col || (unsigned)y_mapped >= (unsigned)row) return 0.0f;
    return src[c * (row * col) + y_mapped * col + x_mapped];
}

template <int kernel_width>
__device__ __forceinline__
float im2col_map_channelwise(const float* __restrict__ src, const int pixel_idx, const int feat_idx, const int row, const int col, const int channels) {
    constexpr int k_sq = kernel_width * kernel_width;
    constexpr int pad = kernel_width / 2;
    const int ic = feat_idx / k_sq;
    const int k = feat_idx % k_sq;
    const int x_mapped = pixel_idx % col - pad + (k % kernel_width);
    const int y_mapped = pixel_idx / col - pad + (k / kernel_width);

    if ((unsigned)x_mapped >= (unsigned)col || (unsigned)y_mapped >= (unsigned)row)
        return 0.0f;
    return src[ic * (row * col) + y_mapped * col + x_mapped];
}

template <int kernel_width, bool real_convolve, bool transpose_weights, bool weight_grad = false>
__global__ void convolve_gemm_kernel(float * __restrict__ dst, const float * __restrict__ src, const float * __restrict__ weights, const int col, const int row, const int in_channels, const int out_channels) {
    __shared__ float s_Weights[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Input[TILE_WIDTH][TILE_WIDTH];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_row = blockIdx.y * TILE_WIDTH + ty;
    const int global_col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;
    
    constexpr int k_sq = kernel_width * kernel_width;
    const int num_pixels = row * col;
    const int reduction_dim = weight_grad ? num_pixels : (in_channels * k_sq);
    const int out_w = weight_grad ? (in_channels * k_sq) : num_pixels;
    const int num_tiles = (reduction_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    #pragma unroll 4
    for (int t = 0; t < num_tiles; ++t) {
        const int tiled_col = t * TILE_WIDTH + tx;
        const int tiled_row = t * TILE_WIDTH + ty;

        // Load A
        if constexpr (weight_grad) {
            s_Weights[ty][tx] = (global_row < out_channels && tiled_col < reduction_dim) 
                ? weights[global_row * reduction_dim + tiled_col] : 0.0f;
        } else if constexpr (transpose_weights) {
            if (global_row < out_channels && tiled_col < reduction_dim) {
                const int oc = tiled_col / k_sq;
                const int k = tiled_col % k_sq;
                s_Weights[ty][tx] = weights[oc * (out_channels * k_sq) + global_row * k_sq + k];
            } else {
                s_Weights[ty][tx] = 0.0f;
            }
        } else {
            s_Weights[ty][tx] = (global_row < out_channels && tiled_col < reduction_dim) 
                ? weights[global_row * reduction_dim + tiled_col] : 0.0f;
        }

        // Load B
        if constexpr (weight_grad) {
            s_Input[ty][tx] = (tiled_row < reduction_dim && global_col < out_w) 
                ? im2col_map_channelwise<kernel_width>(src, tiled_row, global_col, row, col, in_channels) : 0.0f;
        } else {
            s_Input[ty][tx] = (tiled_row < reduction_dim && global_col < out_w) 
                ? im2col_map_coords<kernel_width, real_convolve>(src, tiled_row / k_sq, tiled_row % k_sq, global_col, row, col) : 0.0f;
        }

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) sum += s_Weights[ty][k] * s_Input[k][tx];
        __syncthreads();
    }

    if (global_row < out_channels && global_col < out_w) 
        dst[global_row * out_w + global_col] = sum;
}

__global__ void bias_batched_kernel(float * __restrict__ out, const float * __restrict__ biases, const int n, const int channels)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int c = blockIdx.z;
    if (i < n && c < channels)
        out[c * n + i] += biases[c];
}

template <int TILE_SIZE>
__global__ void reduction_batched_kernel(float * __restrict__ out, const float * __restrict__ in, const int n, const int channels)
{
    __shared__ float tile[TILE_SIZE];
    const int c = blockIdx.z;
    if (c >= channels) return;
    
    const int start = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = c * n;
    float sum = 0.0f;
    
    if (start < n) sum += in[offset + start];
    if (start + blockDim.x < n) sum += in[offset + start + blockDim.x];
    tile[threadIdx.x] = sum;
    __syncthreads();
    
    #pragma unroll
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride)
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        __syncthreads();
    }
    
    // Warp-level reduction using shuffle
    if (threadIdx.x < 32) {
        float val = tile[threadIdx.x];
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(mask, val, offset);
            if (threadIdx.x + offset < blockDim.x) val += other;
        }
        if (threadIdx.x == 0) tile[0] = val;
    }
    
    if (threadIdx.x == 0)
        atomicAdd(&out[c], tile[0]);
}

Conv2DGPU::Conv2DGPU(std::shared_ptr<Layer> prev, int kernel_size, int filters) : m_kernel_size(kernel_size), m_filters(filters)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    cudaMalloc(reinterpret_cast<void **>(&m_output), x * y * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_input), in_x * in_y * in_z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_weights), m_kernel_size * m_kernel_size * in_z * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_biases), z * sizeof(float));
}

Conv2DGPU::~Conv2DGPU() {
    cudaFree(grad_weights);
    cudaFree(grad_biases);
}

std::tuple<int, int, int> Conv2DGPU::dimension() const
{
    auto [prev_x, prev_y, _] = m_prev->dimension();
    return {prev_x, prev_y, m_filters};
}

size_t Conv2DGPU::paramCount() const
{
    return weightCount() + biasCount();
}

size_t Conv2DGPU::weightCount() const
{
    const auto [_, __, prev_z] = m_prev->dimension();
    return m_kernel_size * m_kernel_size * prev_z * m_filters;
}

size_t Conv2DGPU::biasCount() const
{
    return m_filters;
}
void Conv2DGPU::setParams(float *params)
{
    auto [prev_x, prev_y, prev_z] = m_prev->dimension();
    m_weights = params;
    m_biases = params + m_kernel_size * m_kernel_size * prev_z * m_filters;
}
void Conv2DGPU::forward() {
    m_prev->forward();

    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    const int num_pixels = in_w * in_h;
    const int n = out_w * out_h;

    cudaMemset(m_output, 0, n * out_c * sizeof(float));

    constexpr dim3 blockSize{TILE_WIDTH, TILE_WIDTH};
    const dim3 gridSize{
        (unsigned int)((num_pixels + TILE_WIDTH - 1) / TILE_WIDTH),
        (unsigned int)((out_c + TILE_WIDTH - 1) / TILE_WIDTH)
    };

    convolve_gemm_kernel<KERNEL_WIDTH, false, false><<<gridSize, blockSize>>>(
        m_output, m_prev->output(), m_weights, in_w, in_h, in_c, out_c
    );

    constexpr dim3 biasBlockSize{BLOCK_SIZE_1D};
    const dim3 biasGridSize{
        (unsigned int)((n + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D), 1, (unsigned int)out_c
    };

    bias_batched_kernel<<<biasGridSize, biasBlockSize>>>(m_output, m_biases, n, out_c);
}

void Conv2DGPU::backward(const float learning_rate, const float * __restrict__ _grad_output)
{
    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    const int num_pixels = in_w * in_h;
    const int n = out_h * out_w;
    const int k_flat = in_c * m_kernel_size * m_kernel_size;
    
    cudaMemset(grad_input, 0, num_pixels * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, k_flat * out_c * sizeof(float));
    cudaMemset(grad_biases, 0, m_filters * sizeof(float));
    
    constexpr dim3 blockSize{TILE_WIDTH, TILE_WIDTH};
    
    // 1. Grad Input
    const dim3 gridSizeInput{
        (unsigned int)((num_pixels + TILE_WIDTH - 1) / TILE_WIDTH),
        (unsigned int)((in_c + TILE_WIDTH - 1) / TILE_WIDTH)
    };
    convolve_gemm_kernel<KERNEL_WIDTH, true, true><<<gridSizeInput, blockSize>>>(
        grad_input, _grad_output, m_weights, in_w, in_h, out_c, in_c);
    
    // 2. Grad Weights
    const dim3 gridSizeWeights{
        (unsigned int)((k_flat + TILE_WIDTH - 1) / TILE_WIDTH),
        (unsigned int)((out_c + TILE_WIDTH - 1) / TILE_WIDTH)
    };
    convolve_gemm_kernel<KERNEL_WIDTH, false, false, true><<<gridSizeWeights, blockSize>>>(
        grad_weights, m_prev->output(), _grad_output, in_w, in_h, in_c, out_c);
    
    // 3. Grad Biases
    constexpr dim3 biasBlockSize{TILE_WIDTH};
    const dim3 biasGridSize{(unsigned int)((n + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH)), 1, (unsigned int)out_c};
    reduction_batched_kernel<TILE_WIDTH><<<biasGridSize, biasBlockSize>>>(grad_biases, _grad_output, n, out_c);
    
    CHECK(cudaDeviceSynchronize());
    clipGradientsGPU(grad_weights, 5.0f, out_c * k_flat);
    clipGradientsGPU(grad_biases, 5.0f, m_filters);
    updateWeightsGPU(m_weights, grad_weights, learning_rate, out_c * k_flat);
    updateWeightsGPU(m_biases, grad_biases, learning_rate, m_filters);
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, grad_input);
}