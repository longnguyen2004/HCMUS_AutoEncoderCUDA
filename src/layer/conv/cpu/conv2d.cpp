#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <algorithm>

void convolve_cpu(float *dst, const float *src, const float *kernel, int col, int row, int kernel_width, bool real_convolution = false)
{
    int kernel_radius = kernel_width / 2;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            float val = 0;
            for (int k = 0; k < kernel_width; ++k)
            {
                for (int l = 0; l < kernel_width; ++l)
                {
                    int i_mapped = i + k - kernel_radius;
                    int j_mapped = j + l - kernel_radius;

                    if (i_mapped >= 0 && i_mapped < row && j_mapped >= 0 && j_mapped < col)
                    {
                        int k_idx = real_convolution ? (kernel_width - 1 - k) : k;
                        int l_idx = real_convolution ? (kernel_width - 1 - l) : l;
                        val += src[i_mapped * col + j_mapped] * kernel[k_idx * kernel_width + l_idx];
                    }
                }
            }
            dst[i * col + j] += val;
        }
    }
}

void convolve_cpu_backward_kernel(float *grad_kernel, const float *grad_dst, const float *src,
                                  int col, int row, int kernel_width)
{
    int kernel_radius = kernel_width / 2;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            for (int k = 0; k < kernel_width; ++k)
            {
                for (int l = 0; l < kernel_width; ++l)
                {
                    int i_mapped = i + k - kernel_radius;
                    int j_mapped = j + l - kernel_radius;

                    if (i_mapped >= 0 && i_mapped < row && j_mapped >= 0 && j_mapped < col)
                    {
                        grad_kernel[k * kernel_width + l] += grad_dst[i * col + j] * src[i_mapped * col + j_mapped];
                    }
                }
            }
        }
    }
}

Conv2DCPU::Conv2DCPU(std::shared_ptr<Layer> prev, int kernel_size, int filters):
    m_kernel_size(kernel_size), m_filters(filters)
{
    m_prev = prev;
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [x, y, z] = this->dimension();
    m_output.resize(x * y * z);
    grad_input.resize(in_x * in_y * in_z);
    m_grad_weights.resize(m_kernel_size * m_kernel_size * m_filters * in_z);
    m_grad_biases.resize(m_filters);
}
std::tuple<int, int, int> Conv2DCPU::dimension() const
{
    auto [prev_x, prev_y, _] = m_prev->dimension();
    return {prev_x, prev_y, m_filters};
}
size_t Conv2DCPU::paramCount() const
{
    return weightCount() + biasCount();
}

size_t Conv2DCPU::weightCount() const
{
    const auto [_, __, prev_z] = m_prev->dimension();
    return m_kernel_size * m_kernel_size * prev_z * m_filters;
}

size_t Conv2DCPU::biasCount() const
{
    return m_filters;
}

void Conv2DCPU::setParams(float *params)
{
    auto [prev_x, prev_y, prev_z] = m_prev->dimension();
    m_weights = params;
    m_biases = params + m_kernel_size * m_kernel_size * prev_z * m_filters;
}
void Conv2DCPU::forward()
{
    m_prev->forward();
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    std::fill(m_output.begin(), m_output.end(), 0.0f);

    auto in = Kokkos::mdspan(m_prev->output(), in_c, in_w, in_h);
    auto out = Kokkos::mdspan(m_output.data(), out_c, out_w, out_h);
    auto weights = Kokkos::mdspan(m_weights, out_c, in_c, m_kernel_size, m_kernel_size);
    // For each filter
    for (int i = 0; i < out_c; ++i)
    {
        // For each input channel
        for (int j = 0; j < in_c; ++j)
        {
            // Convolve the channel with the kernel for that channel
            convolve_cpu(
                out.data_handle() + out.mapping()(i, 0, 0),
                in.data_handle() + in.mapping()(j, 0, 0),
                weights.data_handle() + weights.mapping()(i, j, 0, 0),
                in_w, in_h, m_kernel_size);
        }
        for (int w = 0; w < out_w; ++w)
            for (int h = 0; h < out_h; ++h)
                out(i, w, h) += m_biases[i];
    }
}

void Conv2DCPU::backward(float learning_rate, const float *grad_output)
{
    auto [in_w, in_h, in_c] = m_prev->dimension();
    auto [out_w, out_h, out_c] = dimension();
    
    // Initialize gradients
    std::fill(this->grad_input.begin(), this->grad_input.end(), 0.0f);
    std::fill(m_grad_weights.begin(), m_grad_weights.end(), 0.0f);
    std::fill(m_grad_biases.begin(), m_grad_biases.end(), 0.0f);

    auto grad_out = Kokkos::mdspan(grad_output, out_c, out_w, out_h);
    auto grad_in = Kokkos::mdspan(grad_input.data(), in_c, in_w, in_h);
    auto in = Kokkos::mdspan(m_prev->output(), in_c, in_w, in_h);
    auto weights = Kokkos::mdspan(m_weights, out_c, in_c, m_kernel_size, m_kernel_size);
    auto grad_weights = Kokkos::mdspan(m_grad_weights.data(), out_c, in_c, m_kernel_size, m_kernel_size);

    // For each filter
    for (int i = 0; i < out_c; ++i)
    {
        // Compute bias gradient
        for (int w = 0; w < out_w; ++w)
            for (int h = 0; h < out_h; ++h)
                m_grad_biases[i] += grad_out(i, w, h);

        // For each input channel
        for (int j = 0; j < in_c; ++j)
        {
            // Gradient with respect to input: convolve grad_out with flipped weights
            convolve_cpu(
                grad_in.data_handle() + grad_in.mapping()(j, 0, 0),
                grad_out.data_handle() + grad_out.mapping()(i, 0, 0),
                weights.data_handle() + weights.mapping()(i, j, 0, 0),
                in_w, in_h, m_kernel_size,
                true  // Use real convolution (flips kernel)
            );

            // Gradient with respect to weights: accumulate src * grad_dst correlations
            convolve_cpu_backward_kernel(
                grad_weights.data_handle() + grad_weights.mapping()(i, j, 0, 0),
                grad_out.data_handle() + grad_out.mapping()(i, 0, 0),
                in.data_handle() + in.mapping()(j, 0, 0),
                out_w, out_h, m_kernel_size
            );
        }
    }

    // Propagate gradients to previous layer
    m_prev->backward(learning_rate, this->grad_input.data());

    // Update parameters
    for (size_t i = 0; i < m_grad_weights.size(); ++i) {
        m_weights[i] -= learning_rate * m_grad_weights[i];
    }
    for (size_t i = 0; i < m_grad_biases.size(); ++i) {
        m_biases[i] -= learning_rate * m_grad_biases[i];
    }
}
