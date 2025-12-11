#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <core/optimizer.h>
#include <cuda_runtime.h>

__global__ void convolve_gpu_kernel(
    float *dst, const float *src, const float *kernel, int col, int row, int kernel_width, bool real_convolve = false)
{
    extern __shared__ float s_src[];
    int kernel_radius = kernel_width / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockStart_x = blockDim.x * blockIdx.x;
    int blockStart_y = blockDim.y * blockIdx.y;
    int x = blockStart_x + tx;
    int y = blockStart_y + ty;

    int tileWidth = blockDim.x + kernel_radius * 2;
    int tileHeight = blockDim.y + kernel_radius * 2;

    for (int j = ty; j < tileHeight; j += blockDim.y)
    {
        for (int i = tx; i < tileWidth; i += blockDim.x)
        {
            int gy = blockStart_y - kernel_radius + j;
            int gx = blockStart_x - kernel_radius + i;
            if (gy < 0 || gy >= row || gx < 0 || gx >= col)
                s_src[j * tileWidth + i] = 0;
            else
                s_src[j * tileWidth + i] = src[gy * col + gx];
        }
    }

    __syncthreads();

    if (x >= col || y >= row)
        return;

    float pixel = 0.0f;
    for (int yf = 0; yf < kernel_width; ++yf)
    {
        for (int xf = 0; xf < kernel_width; ++xf)
        {
            int x_mapped = tx + xf;
            int y_mapped = ty + yf;
            int y_kernel = real_convolve ? (kernel_width - 1 - yf) : yf;
            int x_kernel = real_convolve ? (kernel_width - 1 - xf) : xf;
            pixel += s_src[tileWidth * y_mapped + x_mapped] * kernel[y_kernel * kernel_width + x_kernel];
        }
    }
    if (real_convolve)
        atomicAdd(&dst[col * y + x], pixel);
    else
        dst[col * y + x] += pixel;
}

// Batched version: processes multiple (out_channel, in_channel) pairs in parallel
__global__ void convolve_batched_kernel(
    float *dst, const float *src, const float *weights,
    int col, int row, int kernel_width,
    int in_channels, int out_channels,
    bool real_convolve = false,
    bool transpose_weights = false)
{
    extern __shared__ float s_src[];
    int kernel_radius = kernel_width / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockStart_x = blockDim.x * blockIdx.x;
    int blockStart_y = blockDim.y * blockIdx.y;
    int oc = blockIdx.z;  // Output channel
    int x = blockStart_x + tx;
    int y = blockStart_y + ty;
    
    if (oc >= out_channels) return;

    int tileWidth = blockDim.x + kernel_radius * 2;
    int tileHeight = blockDim.y + kernel_radius * 2;
    int plane_size = col * row;
    int kernel_size = kernel_width * kernel_width;
    
    float result = 0.0f;
    
    // Process all input channels for this output channel
    for (int ic = 0; ic < in_channels; ++ic)
    {
        // Load input tile for this input channel
        for (int j = ty; j < tileHeight; j += blockDim.y)
        {
            for (int i = tx; i < tileWidth; i += blockDim.x)
            {
                int gy = blockStart_y - kernel_radius + j;
                int gx = blockStart_x - kernel_radius + i;
                if (gy < 0 || gy >= row || gx < 0 || gx >= col)
                    s_src[j * tileWidth + i] = 0;
                else
                    s_src[j * tileWidth + i] = src[ic * plane_size + gy * col + gx];
            }
        }
        __syncthreads();
        
        if (x < col && y < row)
        {
            int weight_index = transpose_weights
                                   ? (ic * out_channels + oc)
                                   : (oc * in_channels + ic);
            const float* kernel = weights + weight_index * kernel_size;
            float pixel = 0.0f;
            for (int yf = 0; yf < kernel_width; ++yf)
            {
                for (int xf = 0; xf < kernel_width; ++xf)
                {
                    int x_mapped = tx + xf;
                    int y_mapped = ty + yf;
                    int y_kernel = real_convolve ? (kernel_width - 1 - yf) : yf;
                    int x_kernel = real_convolve ? (kernel_width - 1 - xf) : xf;
                    pixel += s_src[tileWidth * y_mapped + x_mapped] * kernel[y_kernel * kernel_width + x_kernel];
                }
            }
            result += pixel;
        }
        __syncthreads();
    }
    
    if (x < col && y < row)
    {
        if (real_convolve)
            atomicAdd(&dst[oc * plane_size + y * col + x], result);
        else
            dst[oc * plane_size + y * col + x] += result;
    }
}

__global__ void bias_kernel(float *out, const float *bias, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] += *bias;
}

__global__ void bias_batched_kernel(float *out, const float *biases, int n, int channels)
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
            int ix = tx + lx + pad - kw;
            int iy = ty + ly + pad - kh;
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

template<int TILE_H, int TILE_W>
__global__ void grad_weights_kernel(
    float *__restrict__ dW,       // [K*K]
    const float *__restrict__ X,  // [H*W]
    const float *__restrict__ dY, // [H*W]
    int H, int W,
    int K,
    int pad)
{
    // Each block computes ONE (kh, kw)
    int kw = blockIdx.x;
    int kh = blockIdx.y;

    if (kw >= K || kh >= K)
        return;

    // Shared memory tiles
    __shared__ float tileX[TILE_H][TILE_W];
    __shared__ float tileDY[TILE_H][TILE_W];

    float acc = 0.f;

    // Loop over tiles
    for (int ty = 0; ty < H; ty += TILE_H)
    {
        for (int tx = 0; tx < W; tx += TILE_W)
        {

            int lx = threadIdx.x;
            int ly = threadIdx.y;

            // Load X tile ----------------------------------------------------
            int ix = tx + lx + pad - kw;
            int iy = ty + ly + pad - kh;

            if (ix >= 0 && ix < W && iy >= 0 && iy < H)
                tileX[ly][lx] = X[iy * W + ix];
            else
                tileX[ly][lx] = 0.f;

            // Load dY tile ---------------------------------------------------
            int dyx = tx + lx;
            int dyy = ty + ly;

            if (dyx < W && dyy < H)
                tileDY[ly][lx] = dY[dyy * W + dyx];
            else
                tileDY[ly][lx] = 0.f;

            __syncthreads();

            // Compute partial sum from this tile -----------------------------
            for (int i = 0; i < TILE_H; i++)
            {
                for (int j = 0; j < TILE_W; j++)
                {
                    acc += tileX[i][j] * tileDY[i][j];
                }
            }

            __syncthreads();
        }
    }

    // Write result
    dW[kh * K + kw] = acc;
}

template <int TILE_SIZE>
__global__ void reduction_batched_kernel(float *out, const float *in, int n, int channels)
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

template <int TILE_SIZE>
__global__ void reduction_kernel(float *out, const float *in, int n)
{
    __shared__ float tile[TILE_SIZE];
    int start = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (start < n)
        sum += in[start];
    if (start + blockDim.x < n)
        sum += in[start + blockDim.x];
    tile[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, tile[0]);
}

Conv2DGPU::Conv2DGPU(std::shared_ptr<Layer> prev, int kernel_size, int filters) : m_kernel_size(kernel_size), m_filters(filters)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    cudaMalloc(reinterpret_cast<void **>(&m_output), x * y * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_input), in_x * in_y * in_z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_weights), m_kernel_size * m_kernel_size * in_z * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&grad_biases), x * y * z * sizeof(float));
}

Conv2DGPU::~Conv2DGPU() {
    cudaFree(m_output);
    cudaFree(grad_input);
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
void Conv2DGPU::forward()
{
    m_prev->forward();
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    cudaMemset(m_output, 0, out_w * out_h * out_c * sizeof(float));

    // Single batched convolution for all channels
    dim3 blockSize{32, 32};
    dim3 gridSize{
        (in_w + blockSize.x - 1) / blockSize.x,
        (in_h + blockSize.y - 1) / blockSize.y,
        static_cast<unsigned int>(out_c)};  // Z dimension for output channels
    int padding = m_kernel_size / 2;
    size_t smem = (blockSize.x + padding * 2) * (blockSize.y + padding * 2) * sizeof(float);
    
    convolve_batched_kernel<<<gridSize, blockSize, smem>>>(
        m_output, m_prev->output(), m_weights,
        in_w, in_h, m_kernel_size, in_c, out_c, false);
    
    // Add biases - also batched
    int n = out_w * out_h;
    dim3 biasBlockSize{256};
    dim3 biasGridSize{(n + biasBlockSize.x - 1) / biasBlockSize.x, 1, (unsigned int)out_c};
    bias_batched_kernel<<<biasGridSize, biasBlockSize>>>(
        m_output, m_biases, n, out_c);
}
void Conv2DGPU::backward(float learning_rate, const float *_grad_output)
{
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    cudaMemset(grad_input, 0, in_w * in_h * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, m_kernel_size * m_kernel_size * in_c * out_c * sizeof(float));
    cudaMemset(grad_biases, 0, m_filters * sizeof(float));
    
    int padding = m_kernel_size / 2;
    constexpr int TILE_SIZE = 16;
    
    // 1. Batched gradient input computation - all output channels in parallel
    dim3 blockSize{32, 32};
    dim3 gridSize{
        (in_w + blockSize.x - 1) / blockSize.x,
        (in_h + blockSize.y - 1) / blockSize.y,
        static_cast<unsigned int>(in_c)};
    size_t smem = (blockSize.x + padding * 2) * (blockSize.y + padding * 2) * sizeof(float);
    
    convolve_batched_kernel<<<gridSize, blockSize, smem>>>(
        grad_input, _grad_output, m_weights,
        in_w, in_h, m_kernel_size, out_c, in_c, true, true);
    
    // 2. Batched gradient weights computation - all (oc, ic) pairs in parallel
    int total_pairs = out_c * in_c;
    dim3 weightsGrid{(unsigned int)m_kernel_size, (unsigned int)m_kernel_size, (unsigned int)total_pairs};
    dim3 weightsBlock{TILE_SIZE, TILE_SIZE};
    
    grad_weights_batched_kernel<TILE_SIZE, TILE_SIZE><<<weightsGrid, weightsBlock>>>(
        grad_weights, m_prev->output(), _grad_output,
        in_w, in_h, m_kernel_size, padding, in_c, out_c);
    
    // 3. Batched bias gradient reduction - all output channels in parallel
    int n = out_h * out_w;
    dim3 biasBlockSize{TILE_SIZE};
    dim3 biasGridSize{(unsigned int)((n + 2 * TILE_SIZE - 1) / (2 * TILE_SIZE)), 1, (unsigned int)out_c};
    
    reduction_batched_kernel<TILE_SIZE><<<biasGridSize, biasBlockSize>>>(
        grad_biases, _grad_output, n, out_c);
    
    CHECK(cudaDeviceSynchronize());
    clipGradientsGPU(grad_weights, 5.0f, out_c * in_c * m_kernel_size * m_kernel_size);
    clipGradientsGPU(grad_biases, 5.0f, m_filters);
    updateWeightsGPU(m_weights, grad_weights, learning_rate, out_c * in_c * m_kernel_size * m_kernel_size);
    updateWeightsGPU(m_biases, grad_biases, learning_rate, m_filters);
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, grad_input);
}
