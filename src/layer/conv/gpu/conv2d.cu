#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <core/optimizer.h>

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
    atomicAdd(&dst[col * y + x], pixel);
}

__global__ void bias_kernel(float *out, const float *bias, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] += *bias;
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
std::tuple<int, int, int> Conv2DGPU::dimension() const
{
    auto [prev_x, prev_y, _] = m_prev->dimension();
    return {prev_x, prev_y, m_filters};
}
size_t Conv2DGPU::paramCount() const
{
    auto [prev_x, prev_y, prev_z] = m_prev->dimension();
    return m_kernel_size * m_kernel_size * prev_z * m_filters + m_filters;
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

    auto in = Kokkos::mdspan(m_prev->output(), in_c, in_w, in_h);
    auto out = Kokkos::mdspan(m_output, out_c, out_w, out_h);
    auto weights = Kokkos::mdspan(m_weights, out_c, in_c, m_kernel_size, m_kernel_size);

    dim3 blockSize{16, 16};
    dim3 gridSize{
        (in_w + blockSize.x - 1) / blockSize.x,
        (in_h + blockSize.y - 1) / blockSize.y};
    int padding = m_kernel_size / 2;
    // For each filter
    for (int i = 0; i < out_c; ++i)
    {
        // For each input channel
        for (int j = 0; j < in_c; ++j)
        {
            // Convolve the channel with the kernel for that channel
            // TODO: use the unused z part of blockSize for multi-channel calc
            convolve_gpu_kernel<<<gridSize, blockSize, (blockSize.x + padding * 2) * (blockSize.y + padding * 2) * sizeof(float)>>>(
                out.data_handle() + out.mapping()(i, 0, 0),
                in.data_handle() + in.mapping()(j, 0, 0),
                weights.data_handle() + weights.mapping()(i, j, 0, 0),
                in_w, in_h, m_kernel_size);
        }
        int n = out_w * out_h;
        dim3 biasBlockSize{32};
        dim3 biasGridSize{(n + biasBlockSize.x - 1) / biasBlockSize.x};
        bias_kernel<<<biasGridSize, biasBlockSize>>>(
            out.data_handle() + out.mapping()(i, 0, 0),
            m_biases + i, n);
    }
    CHECK(cudaDeviceSynchronize());
}
void Conv2DGPU::backward(float learning_rate, const float *_grad_output)
{
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    cudaMemset(grad_input, 0, in_w * in_h * in_c * sizeof(float));
    cudaMemset(grad_weights, 0, m_kernel_size * m_kernel_size * in_c * out_c * sizeof(float));
    cudaMemset(grad_biases, 0, m_filters * sizeof(float));
    auto in = Kokkos::mdspan(m_prev->output(), in_c, in_w, in_h);
    auto grad_output = Kokkos::mdspan(_grad_output, out_c, out_w, out_h);
    auto grad_input = Kokkos::mdspan(this->grad_input, in_c, in_w, in_h);
    auto weights = Kokkos::mdspan(m_weights, out_c, in_c, m_kernel_size, m_kernel_size);
    auto grad_weights = Kokkos::mdspan(this->grad_weights, out_c, in_c, m_kernel_size, m_kernel_size);
    dim3 blockSize{16, 16};
    dim3 gridSize{
        (in_w + blockSize.x - 1) / blockSize.x,
        (in_h + blockSize.y - 1) / blockSize.y};
    int padding = m_kernel_size / 2;

    // For each filter
    for (int oc = 0; oc < out_c; ++oc)
    {
        constexpr int TILE_SIZE = 16;
        // For each input channel
        for (int ic = 0; ic < in_c; ++ic)
        {
            dim3 grad_gridSize{
                (out_w + blockSize.x - 1) / blockSize.x,
                (out_h + blockSize.y - 1) / blockSize.y};
            convolve_gpu_kernel<<<grad_gridSize, blockSize, (blockSize.x + padding * 2) * (blockSize.y + padding * 2) * sizeof(float)>>>(
                grad_input.data_handle() + grad_input.mapping()(ic, 0, 0),
                grad_output.data_handle() + grad_output.mapping()(oc, 0, 0),
                weights.data_handle() + weights.mapping()(oc, ic, 0, 0),
                out_w, out_h, m_kernel_size, true);
            grad_weights_kernel<TILE_SIZE, TILE_SIZE><<<dim3(m_kernel_size, m_kernel_size), dim3(TILE_SIZE, TILE_SIZE)>>>(
                grad_weights.data_handle() + grad_weights.mapping()(oc, ic, 0, 0),
                in.data_handle() + in.mapping()(ic, 0, 0),
                grad_output.data_handle() + grad_output.mapping()(oc, 0, 0),
                in_w, in_h, m_kernel_size, padding);
        }
        dim3 blockSize{TILE_SIZE};
        dim3 gridSize{(m_filters + TILE_SIZE - 1u) / TILE_SIZE};
        reduction_kernel<TILE_SIZE><<<gridSize, blockSize>>>(
            grad_biases + oc, grad_output.data_handle() + grad_output.mapping()(oc, 0, 0), out_h * out_w
        );
    }
    updateWeightsGPU(m_weights, this->grad_weights, learning_rate, out_c * in_c * m_kernel_size * m_kernel_size);
    updateWeightsGPU(m_biases, this->grad_biases, learning_rate, m_filters);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    m_prev->backward(learning_rate, this->grad_input);
}
