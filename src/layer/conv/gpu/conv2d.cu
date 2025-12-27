#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <core/optimizer.h>
#include <cuda_runtime.h>
#include <constants.h>
#include <helper/gpu_helper.h>

// Implementation:
// 1. Naive: no shared memory
// 2. Optimized naive: shared memory + tree reduction for grad weights
// 3. im2col

#define IMPLEMENTATION 3

__global__ void naive_conv_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    const int width,
    const int height,
    const int in_channels,
    const int out_channels,
    const int kernel_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int f = blockIdx.z;

    if (x < width && y < height && f < out_channels) {
        float val = biases[f];
        const int pad = kernel_size / 2;
        for (int c = 0; c < in_channels; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int in_x = x + kx - pad;
                    const int in_y = y + ky - pad;
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        val += input[c * (height * width) + in_y * width + in_x] *
                               weights[((f * in_channels + c) * kernel_size + ky) * kernel_size + kx];
                    }
                }
            }
        }
        output[f * (height * width) + y * width + x] = val;
    }
}

__global__ void naive_conv_backward_input_kernel(
    float* __restrict__ grad_input,
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    const int width,
    const int height,
    const int in_channels,
    const int out_channels,
    const int kernel_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;

    if (x < width && y < height && c < in_channels) {
        float val = 0.0f;
        const int pad = kernel_size / 2;
        for (int f = 0; f < out_channels; ++f) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int out_x = x + kx - pad;
                    const int out_y = y + ky - pad;
                    if (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height) {
                        const int ky_flipped = kernel_size - 1 - ky;
                        const int kx_flipped = kernel_size - 1 - kx;
                        val += grad_output[f * (height * width) + out_y * width + out_x] *
                               weights[((f * in_channels + c) * kernel_size + ky_flipped) * kernel_size + kx_flipped];
                    }
                }
            }
        }
        grad_input[c * (height * width) + y * width + x] = val;
    }
}

__global__ void naive_conv_backward_weights_kernel(
    float* __restrict__ grad_weights,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const int width,
    const int height,
    const int in_channels,
    const int out_channels,
    const int kernel_size)
{
    const int kx = blockIdx.x;
    const int ky = blockIdx.y;
    const int fc = blockIdx.z;
    
    const int f = fc / in_channels;
    const int c = fc % in_channels;

    if (f < out_channels && c < in_channels && ky < kernel_size && kx < kernel_size) {
        float val = 0.0f;
        const int pad = kernel_size / 2;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int in_x = x + kx - pad;
                const int in_y = y + ky - pad;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    val += grad_output[f * (height * width) + y * width + x] *
                           input[c * (height * width) + in_y * width + in_x];
                }
            }
        }
        grad_weights[((f * in_channels + c) * kernel_size + ky) * kernel_size + kx] = val;
    }
}

__global__ void naive_conv_backward_biases_kernel(
    float* __restrict__ grad_biases,
    const float* __restrict__ grad_output,
    const int width,
    const int height,
    const int out_channels)
{
    const int f = blockIdx.x;
    if (f < out_channels) {
        float val = 0.0f;
        const int num_pixels = width * height;
        for (int i = 0; i < num_pixels; ++i) {
            val += grad_output[f * num_pixels + i];
        }
        grad_biases[f] = val;
    }
}

template <int KERNEL_SIZE, int TILE_SIZE, int CHANNELS_PER_BLOCK, bool BACKWARD_MODE>
__global__ void simple_conv_unified_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    const int width,
    const int height,
    const int in_channels,
    const int out_channels)
{
    // Shared memory for input tile (with padding) and weights
    __shared__ float s_input[CHANNELS_PER_BLOCK][TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    __shared__ float s_weights[CHANNELS_PER_BLOCK][KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_x = blockIdx.x * TILE_SIZE + tx;
    const int out_y = blockIdx.y * TILE_SIZE + ty;
    const int out_c = blockIdx.z;

    const bool valid_output = (out_x < width && out_y < height && out_c < out_channels);
    
    const int pad = KERNEL_SIZE / 2;
    float sum = 0.0f;
    
    // Loop over input channels in batches
    for (int ic_base = 0; ic_base < in_channels; ic_base += CHANNELS_PER_BLOCK) {
        const int channels_in_batch = min(CHANNELS_PER_BLOCK, in_channels - ic_base);
        
        // Load input tile into shared memory - ALL threads participate
        for (int c = 0; c < channels_in_batch; ++c) {
            const int ic = ic_base + c;
            
            // Load main tile - each thread loads its corresponding position
            {
                const int in_x = blockIdx.x * TILE_SIZE + tx - pad;
                const int in_y = blockIdx.y * TILE_SIZE + ty - pad;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    s_input[c][ty][tx] = input[ic * (height * width) + in_y * width + in_x];
                } else {
                    s_input[c][ty][tx] = 0.0f;
                }
            }
            
            // Load right halo region (only threads with tx < KERNEL_SIZE - 1)
            if (tx < KERNEL_SIZE - 1) {
                const int in_x = blockIdx.x * TILE_SIZE + TILE_SIZE + tx - pad;
                const int in_y = blockIdx.y * TILE_SIZE + ty - pad;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    s_input[c][ty][TILE_SIZE + tx] = input[ic * (height * width) + in_y * width + in_x];
                } else {
                    s_input[c][ty][TILE_SIZE + tx] = 0.0f;
                }
            }
            
            // Load bottom halo region (only threads with ty < KERNEL_SIZE - 1)
            if (ty < KERNEL_SIZE - 1) {
                const int in_x = blockIdx.x * TILE_SIZE + tx - pad;
                const int in_y = blockIdx.y * TILE_SIZE + TILE_SIZE + ty - pad;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    s_input[c][TILE_SIZE + ty][tx] = input[ic * (height * width) + in_y * width + in_x];
                } else {
                    s_input[c][TILE_SIZE + ty][tx] = 0.0f;
                }
            }
            
            // Load bottom-right corner halo region
            if (tx < KERNEL_SIZE - 1 && ty < KERNEL_SIZE - 1) {
                const int in_x = blockIdx.x * TILE_SIZE + TILE_SIZE + tx - pad;
                const int in_y = blockIdx.y * TILE_SIZE + TILE_SIZE + ty - pad;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    s_input[c][TILE_SIZE + ty][TILE_SIZE + tx] = input[ic * (height * width) + in_y * width + in_x];
                } else {
                    s_input[c][TILE_SIZE + ty][TILE_SIZE + tx] = 0.0f;
                }
            }
        }
        
        // Load weights into shared memory
        if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
            for (int c = 0; c < channels_in_batch; ++c) {
                const int ic = ic_base + c;
                if constexpr (BACKWARD_MODE) {
                    // For backward pass: out_c is the input channel, ic is the output channel
                    const int ky_flip = KERNEL_SIZE - 1 - ty;
                    const int kx_flip = KERNEL_SIZE - 1 - tx;
                    const int weight_idx = ((ic * out_channels + out_c) * KERNEL_SIZE + ky_flip) * KERNEL_SIZE + kx_flip;
                    s_weights[c][ty][tx] = weights[weight_idx];
                } else {
                    // Normal weight indexing for forward pass
                    const int weight_idx = ((out_c * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx;
                    s_weights[c][ty][tx] = weights[weight_idx];
                }
            }
        }
        
        __syncthreads();
        
        // Compute convolution for this batch of channels
        if (valid_output) {
            for (int c = 0; c < channels_in_batch; ++c) {
                #pragma unroll
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    #pragma unroll
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        sum += s_input[c][ty + ky][tx + kx] * s_weights[c][ky][kx];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Add bias and write output
    if (valid_output) {
        if constexpr (!BACKWARD_MODE) {
            sum += biases[out_c];
        }
        output[out_c * (height * width) + out_y * width + out_x] = sum;
    }
}

// Backward convolution for gradient w.r.t. weights
template <int KERNEL_SIZE, int TILE_SIZE>
__global__ void simple_conv_backward_weights_kernel(
    float* __restrict__ grad_weights,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const int width,
    const int height,
    const int in_channels,
    const int out_channels)
{
    // Each block computes one weight element across all spatial positions
    const int out_c = blockIdx.z / in_channels;
    const int in_c = blockIdx.z % in_channels;
    const int ky = blockIdx.y;
    const int kx = blockIdx.x;
    
    if (out_c >= out_channels || in_c >= in_channels || ky >= KERNEL_SIZE || kx >= KERNEL_SIZE)
        return;
    
    __shared__ float s_partial[TILE_SIZE * TILE_SIZE];
    
    const int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int num_threads = TILE_SIZE * TILE_SIZE;
    const int pad = KERNEL_SIZE / 2;
    const int num_pixels = width * height;
    
    float thread_sum = 0.0f;
    
    // Each thread processes multiple pixels
    for (int pixel = tid; pixel < num_pixels; pixel += num_threads) {
        const int out_x = pixel % width;
        const int out_y = pixel / width;
        
        // Corresponding input position
        const int in_x = out_x + kx - pad;
        const int in_y = out_y + ky - pad;
        
        float grad_val = grad_output[out_c * num_pixels + pixel];
        float input_val = 0.0f;
        
        if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
            input_val = input[in_c * num_pixels + in_y * width + in_x];
        }
        
        thread_sum += input_val * grad_val;
    }
    
    s_partial[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = num_threads / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        float val = s_partial[tid];
        if (num_threads > 32) val += s_partial[tid + 32];
        
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        
        if (tid == 0) {
            const int weight_idx = ((out_c * in_channels + in_c) * KERNEL_SIZE + ky) * KERNEL_SIZE + kx;
            grad_weights[weight_idx] = val;
        }
    }
}

// Backward convolution for gradient w.r.t. biases
template <int BLOCK_SIZE>
__global__ void simple_conv_backward_biases_kernel(
    float* __restrict__ grad_biases,
    const float* __restrict__ grad_output,
    const int width,
    const int height,
    const int out_channels)
{
    __shared__ float s_sum[BLOCK_SIZE];
    
    const int out_c = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_pixels = width * height;
    
    if (out_c >= out_channels)
        return;
    
    float sum = 0.0f;
    
    // Each thread accumulates a subset of pixels
    for (int i = tid; i < num_pixels; i += blockDim.x) {
        sum += grad_output[out_c * num_pixels + i];
    }
    
    s_sum[tid] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        float val = s_sum[tid];
        if (blockDim.x > 32) val += s_sum[tid + 32];
        
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        
        if (tid == 0) {
            grad_biases[out_c] = val;
        }
    }
}

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

#if IMPLEMENTATION == 1
void Conv2DGPU::forward() {
    m_prev->forward();
    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((out_w + TILE_WIDTH - 1) / TILE_WIDTH, (out_h + TILE_WIDTH - 1) / TILE_WIDTH, out_c);
    
    naive_conv_forward_kernel<<<gridDim, blockDim>>>(
        m_output, m_prev->output(), m_weights, m_biases, in_w, in_h, in_c, out_c, m_kernel_size
    );
    CHECK(cudaGetLastError());
}

void Conv2DGPU::backward(const float learning_rate, const float * __restrict__ _grad_output) {
    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    
    cudaMemset(grad_input, 0, in_w * in_h * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, out_c * in_c * m_kernel_size * m_kernel_size * sizeof(float));
    cudaMemset(grad_biases, 0, out_c * sizeof(float));

    dim3 blockDim2D(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDimInput((in_w + TILE_WIDTH - 1) / TILE_WIDTH, (in_h + TILE_WIDTH - 1) / TILE_WIDTH, in_c);
    naive_conv_backward_input_kernel<<<gridDimInput, blockDim2D>>>(
        grad_input, _grad_output, m_weights, in_w, in_h, in_c, out_c, m_kernel_size
    );

    dim3 gridDimWeights(m_kernel_size, m_kernel_size, out_c * in_c);
    naive_conv_backward_weights_kernel<<<gridDimWeights, 1>>>(
        grad_weights, _grad_output, m_prev->output(), in_w, in_h, in_c, out_c, m_kernel_size
    );

    naive_conv_backward_biases_kernel<<<out_c, 1>>>(
        grad_biases, _grad_output, in_w, in_h, out_c
    );

    CHECK(cudaDeviceSynchronize());
    clipGradientsGPU(grad_weights, 5.0f, out_c * in_c * m_kernel_size * m_kernel_size);
    clipGradientsGPU(grad_biases, 5.0f, out_c);
    updateWeightsGPU(m_weights, grad_weights, learning_rate, out_c * in_c * m_kernel_size * m_kernel_size);
    updateWeightsGPU(m_biases, grad_biases, learning_rate, out_c);
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, grad_input);
}
#elif IMPLEMENTATION == 3
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
#else
void Conv2DGPU::forward() {
    m_prev->forward();

    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    
    constexpr int TILE_SIZE = 16;
    constexpr int CHANNELS_PER_BLOCK = 4;
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (in_w + TILE_SIZE - 1) / TILE_SIZE,
        (in_h + TILE_SIZE - 1) / TILE_SIZE,
        out_c
    );
    
    if (m_kernel_size == 3) {
        simple_conv_unified_kernel<3, TILE_SIZE, CHANNELS_PER_BLOCK, false><<<gridDim, blockDim>>>(
            m_output, m_prev->output(), m_weights, m_biases, in_w, in_h, in_c, out_c
        );
    } else if (m_kernel_size == 5) {
        simple_conv_unified_kernel<5, TILE_SIZE, CHANNELS_PER_BLOCK, false><<<gridDim, blockDim>>>(
            m_output, m_prev->output(), m_weights, m_biases, in_w, in_h, in_c, out_c
        );
    }
    
    CHECK(cudaGetLastError());
}

void Conv2DGPU::backward(const float learning_rate, const float * __restrict__ _grad_output)
{
    const auto [in_w, in_h, in_c] = m_prev->dimension();
    const auto [out_w, out_h, out_c] = dimension();
    
    // Initialize gradient buffers to zero
    cudaMemset(grad_input, 0, in_w * in_h * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, out_c * in_c * m_kernel_size * m_kernel_size * sizeof(float));
    cudaMemset(grad_biases, 0, out_c * sizeof(float));
    
    constexpr int TILE_SIZE = 16;
    constexpr int CHANNELS_PER_BLOCK = 4;
    
    // 1. Gradient w.r.t. input using unified kernel in backward mode
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim(
            (in_w + TILE_SIZE - 1) / TILE_SIZE,
            (in_h + TILE_SIZE - 1) / TILE_SIZE,
            in_c
        );
        
        if (m_kernel_size == 3) {
            simple_conv_unified_kernel<3, TILE_SIZE, CHANNELS_PER_BLOCK, true><<<gridDim, blockDim>>>(
                grad_input, _grad_output, m_weights, nullptr, in_w, in_h, out_c, in_c
            );
        } else if (m_kernel_size == 5) {
            simple_conv_unified_kernel<5, TILE_SIZE, CHANNELS_PER_BLOCK, true><<<gridDim, blockDim>>>(
                grad_input, _grad_output, m_weights, nullptr, in_w, in_h, out_c, in_c
            );
        }
    }
    
    // 2. Gradient w.r.t. weights using parallel multiplication + tree reduction
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim(m_kernel_size, m_kernel_size, out_c * in_c);
        
        if (m_kernel_size == 3) {
            simple_conv_backward_weights_kernel<3, TILE_SIZE><<<gridDim, blockDim>>>(
                grad_weights, _grad_output, m_prev->output(), in_w, in_h, in_c, out_c
            );
        } else if (m_kernel_size == 5) {
            simple_conv_backward_weights_kernel<5, TILE_SIZE><<<gridDim, blockDim>>>(
                grad_weights, _grad_output, m_prev->output(), in_w, in_h, in_c, out_c
            );
        }
    }
    
    // 3. Gradient w.r.t. biases using tree reduction
    {
        constexpr int BLOCK_SIZE = 256;
        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim(out_c);
        
        simple_conv_backward_biases_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(
            grad_biases, _grad_output, in_w, in_h, out_c
        );
    }
    
    CHECK(cudaDeviceSynchronize());
    clipGradientsGPU(grad_weights, 5.0f, out_c * in_c * m_kernel_size * m_kernel_size);
    clipGradientsGPU(grad_biases, 5.0f, m_filters);
    updateWeightsGPU(m_weights, grad_weights, learning_rate, out_c * in_c * m_kernel_size * m_kernel_size);
    updateWeightsGPU(m_biases, grad_biases, learning_rate, m_filters);
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, grad_input);
}
#endif
