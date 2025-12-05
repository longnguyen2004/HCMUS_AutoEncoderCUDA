#include "../maxpool2d.h"
#include <mdspan/mdspan.hpp>
#include <algorithm>

MaxPool2DCPU::MaxPool2DCPU(std::shared_ptr<Layer>prev, int stride): m_prev(prev), m_stride(stride) {
    auto [x, y, z] = dimension();
    this->m_output.resize(x * y * z);
}

std::tuple<int, int, int> MaxPool2DCPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {
        (x + m_stride - 1) / m_stride,
        (y + m_stride - 1) / m_stride,
        z
    };
}

void MaxPool2DCPU::forward() {
    m_prev->forward();
    std::fill(m_output.begin(), m_output.end(), -INFINITY);
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [out_x, out_y, out_z] = dimension();
    auto input = Kokkos::mdspan(m_prev->output(), in_z, in_y, in_x);
    auto output = Kokkos::mdspan(m_output.data(), out_z, out_y, out_x);
    for (int c = 0; c < out_z; ++c)
    {
        for (int i = 0; i < in_y; ++i)
        {
            for (int j = 0; j < in_x; ++j)
            {
                int i_mapped = i / m_stride;
                int j_mapped = j / m_stride;
                output(c, i_mapped, j_mapped) = std::fmaxf(output(c, i_mapped, j_mapped), input(c, i, j));
            }
        }
    }
}

void MaxPool2DCPU::backward(float learning_rate, const float* grad_output) {
    auto [in_x, in_y, in_z] = m_prev->dimension();
    auto [out_x, out_y, out_z] = dimension();
    std::vector<float> grad_input(in_x * in_y * in_z);
    auto input_3d = Kokkos::mdspan(m_prev->output(), in_z, in_y, in_x);
    auto output_3d = Kokkos::mdspan(output(), out_z, out_y, out_x);
    auto grad_input_3d = Kokkos::mdspan(grad_input.data(), in_z, in_y, in_x);
    auto grad_output_3d = Kokkos::mdspan(grad_output, out_z, out_y, out_x);
    for (int c = 0; c < out_z; ++c)
    {
        for (int i = 0; i < in_y; ++i)
        {
            for (int j = 0; j < in_x; ++j)
            {
                int i_mapped = i / m_stride;
                int j_mapped = j / m_stride;
                grad_input_3d(c, i, j) =
                    input_3d(c, i, j) == output_3d(c, i_mapped, j_mapped)
                        ? grad_output_3d(c, i_mapped, j_mapped) : 0; 
            }
        }
    }
    this->m_prev->backward(learning_rate, grad_input.data());
}

const float* MaxPool2DCPU::output() const {
    return m_output.data();
}