#include "../conv2d.h"
#include <mdspan/mdspan.hpp>

__global__ void convolve_gpu_kernel(float *dst, const float *src, const float *kernel, int col, int row, int kernel_width)
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
            pixel += s_src[tileWidth * y_mapped + x_mapped] * kernel[yf * kernel_width + xf];
        }
    }
    dst[col * y + x] += pixel;
}

Conv2DGPU::Conv2DGPU(std::shared_ptr<Layer> prev, int kernel_size, int filters) : m_prev(prev), m_kernel_size(kernel_size), m_filters(filters)
{
    auto [x, y, z] = this->dimension();
    cudaMalloc(reinterpret_cast<void **>(&m_output), x * y * z * sizeof(float));
}
Conv2DGPU::~Conv2DGPU()
{
    cudaFree(m_output);
}
std::tuple<int, int, int> Conv2DGPU::dimension() const
{
    auto [prev_x, prev_y, _] = m_prev->dimension();
    return {prev_x, prev_y, m_kernel_size};
}
const float *Conv2DGPU::output() const
{
    return m_output;
}
size_t Conv2DGPU::paramsCount() const
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
    dim3 blockSize{32, 32};
    dim3 gridSize{
        (in_w + blockSize.x - 1) / blockSize.x,
        (in_h + blockSize.y - 1) / blockSize.y};
    // For each filter
    for (int i = 0; i < out_c; ++i)
    {
        // For each input channel
        for (int j = 0; j < in_c; ++j)
        {
            // Convolve the channel with the kernel for that channel
            convolve_gpu_kernel<<<gridSize, blockSize>>>(
                out.data_handle() + out.mapping()(i, 0, 0),
                in.data_handle() + in.mapping()(j, 0, 0),
                weights.data_handle() + weights.mapping()(i, j, 0, 0),
                in_w, in_h, m_kernel_size
            );
        }
        // TODO: add bias
    }
    // TODO: sync
}
