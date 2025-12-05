#include "../conv2d.h"
#include <mdspan/mdspan.hpp>
#include <algorithm>

void convolve_cpu(float* dst, const float* src, const float* kernel, int col, int row, int kernel_width) {
  int kernel_radius = kernel_width / 2;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      float val = 0;
      for (int k = 0; k < kernel_width; ++k) {
        for (int l = 0; l < kernel_width; ++l) {
          int i_mapped = i + k - kernel_radius;
          int j_mapped = j + l - kernel_radius;

          if (i_mapped >= 0 && i_mapped < row && j_mapped >= 0 && j_mapped < col) {
            val += src[i_mapped * col + j_mapped] * kernel[k * kernel_width + l];
          }
        }
      }
      dst[i * col + j] += val;
    }
  }
}


Conv2DCPU::Conv2DCPU(std::shared_ptr<Layer> prev, int kernel_size, int filters) : m_kernel_size(kernel_size), m_filters(filters)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    m_output.resize(x * y * z);
    grad_input.resize(x * y * z);
}
std::tuple<int, int, int> Conv2DCPU::dimension() const
{
    auto [prev_x, prev_y, _] = m_prev->dimension();
    return {prev_x, prev_y, m_filters};
}
size_t Conv2DCPU::paramCount() const
{
    auto [prev_x, prev_y, prev_z] = m_prev->dimension();
    return m_kernel_size * m_kernel_size * prev_z * m_filters + m_filters;
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
                in_w, in_h, m_kernel_size
            );
        }
        // TODO: add bias
    }
}
