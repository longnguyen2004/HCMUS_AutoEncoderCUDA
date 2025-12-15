#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <core/optimizer.h>
#include <cuda_runtime.h>
#include <constants.h>
#include <helper/gpu_helper.h>

template <int kernel_width, bool flip_kernel>
__device__ __forceinline__
float im2col_map(const float* src, int c, int y, int x, int row, int col) {
    if (flip_kernel) y = kernel_width * kernel_width - 1 - y;
    int x_mapped = (x % col) - kernel_width / 2 + (y % kernel_width);
    int y_mapped = (x / col) - kernel_width / 2 + (y / kernel_width);
    if ((unsigned)x_mapped >= (unsigned)col || (unsigned)y_mapped >= (unsigned)row) return 0.0f;
    return src[c * (row * col) + y_mapped * col + x_mapped];
}

template <int kernel_width, bool real_convolve, bool transpose_weights>
__global__ void convolve_gemm_kernel(float * __restrict__ dst, const float * __restrict__ src, const float * __restrict__ weights, int width, int height, int in_channels, int out_channels) {
    __shared__ float s_Weights[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Input[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int global_row = blockIdx.y * TILE_WIDTH + ty;
    int global_col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;
    int num_pixels = height * width;
    int reduction_dim = in_channels * (kernel_width * kernel_width);
    int num_tiles = (reduction_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col = t * TILE_WIDTH + tx;
        int tiled_row = t * TILE_WIDTH + ty;

        if (global_row < out_channels && tiled_col < reduction_dim) {
            if (transpose_weights) {
                int k_sq = kernel_width * kernel_width;
                int oc = tiled_col / k_sq;
                int k = tiled_col % k_sq;
                int ic = global_row;
                int weight_idx = oc * (out_channels * k_sq) + ic * k_sq + k;
                s_Weights[ty][tx] = weights[weight_idx];
            } else {
                s_Weights[ty][tx] = weights[global_row * reduction_dim + tiled_col];
            }
        } else {
            s_Weights[ty][tx] = 0.0f;
        }

        s_Input[ty][tx] = (tiled_row < reduction_dim && global_col < num_pixels) ? im2col_map<kernel_width, real_convolve>(src, tiled_row / (kernel_width * kernel_width), tiled_row % (kernel_width * kernel_width), global_col, height, width) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) sum += s_Weights[ty][k] * s_Input[k][tx];
        __syncthreads();
    }

    if (global_row < out_channels && global_col < num_pixels) dst[global_row * num_pixels + global_col] = sum;
}

__global__ void bias_batched_kernel(float * __restrict__ out, const float * __restrict__ biases, int n, int channels)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockIdx.z;
    if (i < n && c < channels)
        out[c * n + i] += biases[c];
}

template<int TILE_H, int TILE_W>
__global__ void grad_weights_batched_kernel(
    float *__restrict__ dW,       // [out_c * in_c * K * K]
    const float *__restrict__ X,  // [in_c * H * W]
    const float *__restrict__ dY, // [out_c * H * W]
    int H, int W, int K, int pad, int in_c, int out_c)
{
    int kw = blockIdx.x;
    int kh = blockIdx.y;
    int pair_idx = blockIdx.z;  // Linear index for (oc, ic) pairs
    
    if (kw >= K || kh >= K) return;
    
    int oc = pair_idx / in_c;
    int ic = pair_idx % in_c;
    
    if (oc >= out_c || ic >= in_c) return;

    float thread_acc = 0.f;
    int plane_size = H * W;

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    for (int ty = 0; ty < H; ty += TILE_H)
    {
        for (int tx = 0; tx < W; tx += TILE_W)
        {
            // Load X value
            int ix = tx + lx - pad + kw;
            int iy = ty + ly - pad + kh;
            float valX = 0.f;

            if (ix >= 0 && ix < W && iy >= 0 && iy < H)
                valX = X[ic * plane_size + iy * W + ix];

            // Load dY value
            int dyx = tx + lx;
            int dyy = ty + ly;
            float valDY = 0.f;

            if (dyx < W && dyy < H)
                valDY = dY[oc * plane_size + dyy * W + dyx];

            thread_acc += valX * valDY;
        }
    }

    // Reduction within the block (shared memory down to warp, then warp shuffle)
    __shared__ float sdata[TILE_H * TILE_W];
    int tid = ly * TILE_W + lx;
    sdata[tid] = thread_acc;
    __syncthreads();

    int num_threads = TILE_H * TILE_W;
    for (int s = num_threads / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float val = sdata[tid];
    if (tid < 32) {
        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (tid == 0) {
            dW[(oc * in_c + ic) * K * K + kh * K + kw] = val;
        }
    }
}

template <int TILE_SIZE>
__global__ void reduction_batched_kernel(float * __restrict__ out, const float * __restrict__ in, int n, int channels)
{
    __shared__ float tile[TILE_SIZE];
    int c = blockIdx.z;
    if (c >= channels) return;
    
    int start = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    int offset = c * n;
    if (start < n)
        sum += in[offset + start];
    if (start + blockDim.x < n)
        sum += in[offset + start + blockDim.x];
    tile[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        __syncthreads();
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
    auto [prev_x, prev_y, prev_z] = m_prev->dimension();
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

    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();

    cudaMemset(m_output, 0, out_w * out_h * out_c * sizeof(float));

    int num_pixels = in_w * in_h;

    dim3 blockSize{TILE_WIDTH, TILE_WIDTH};
    dim3 gridSize{
        (unsigned int)((num_pixels + blockSize.x - 1) / blockSize.x),
        (unsigned int)((out_c + blockSize.y - 1) / blockSize.y)
    };

    convolve_gemm_kernel<KERNEL_WIDTH, false, false><<<gridSize, blockSize>>>(
        m_output,
        m_prev->output(),
        m_weights,
        in_w, in_h,
        in_c, out_c
    );

    int n = out_w * out_h;
    dim3 biasBlockSize{BLOCK_SIZE_1D};
    dim3 biasGridSize{
        (unsigned int)((n + biasBlockSize.x - 1) / biasBlockSize.x),
        1,
        (unsigned int)out_c
    };

    bias_batched_kernel<<<biasGridSize, biasBlockSize>>>(m_output, m_biases, n, out_c);
}

void Conv2DGPU::backward(float learning_rate, const float *_grad_output)
{
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    cudaMemset(grad_input, 0, in_w * in_h * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, m_kernel_size * m_kernel_size * in_c * out_c * sizeof(float));
    cudaMemset(grad_biases, 0, m_filters * sizeof(float));
    
    // 1. Gradient input computation using GEMM kernel
    int num_pixels = in_w * in_h;
    dim3 blockSize{TILE_WIDTH, TILE_WIDTH};
    dim3 gridSize{
        (unsigned int)((num_pixels + blockSize.x - 1) / blockSize.x),
        (unsigned int)((in_c + blockSize.y - 1) / blockSize.y)
    };
    
    convolve_gemm_kernel<KERNEL_WIDTH, true, true><<<gridSize, blockSize>>>(
        grad_input, _grad_output, m_weights,
        in_w, in_h,
        out_c, in_c);
    
    // 2. Batched gradient weights computation - all (oc, ic) pairs in parallel
    int total_pairs = out_c * in_c;
    dim3 weightsGrid{(unsigned int)m_kernel_size, (unsigned int)m_kernel_size, (unsigned int)total_pairs};
    dim3 weightsBlock{TILE_WIDTH, TILE_WIDTH};
    
    grad_weights_batched_kernel<TILE_WIDTH, TILE_WIDTH><<<weightsGrid, weightsBlock>>>(
        grad_weights, m_prev->output(), _grad_output,
        in_w, in_h, m_kernel_size, KERNEL_RADIUS, in_c, out_c);
    
    // 3. Batched bias gradient reduction - all output channels in parallel
    int n = out_h * out_w;
    dim3 biasBlockSize{TILE_WIDTH};
    dim3 biasGridSize{(unsigned int)((n + 2 * TILE_WIDTH - 1) / (2 * TILE_WIDTH)), 1, (unsigned int)out_c};
    
    reduction_batched_kernel<TILE_WIDTH><<<biasGridSize, biasBlockSize>>>(
        grad_biases, _grad_output, n, out_c);
    
    CHECK(cudaDeviceSynchronize());
    clipGradientsGPU(grad_weights, 5.0f, out_c * in_c * m_kernel_size * m_kernel_size);
    clipGradientsGPU(grad_biases, 5.0f, m_filters);
    updateWeightsGPU(m_weights, grad_weights, learning_rate, out_c * in_c * m_kernel_size * m_kernel_size);
    updateWeightsGPU(m_biases, grad_biases, learning_rate, m_filters);
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, grad_input);
}
