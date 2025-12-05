#include "../upsample2d.h"
#include <constants.h>
#include <mdspan/mdspan.hpp>
#include <cstring>

UpSample2DCPU::UpSample2DCPU(std::shared_ptr<Layer> prev)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    m_output.resize(x * y * z);
    grad_input.resize(in_x * in_y * in_z);
}

std::tuple<int, int, int> UpSample2DCPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {x * SCALE_FACTOR, y * SCALE_FACTOR, z};
}

void UpSample2DCPU::forward()
{
    m_prev->forward();
    auto [x, y, z] = dimension();
    auto [x_in, y_in, _] = m_prev->dimension();

    auto output = Kokkos::mdspan(m_output.data(), z, y, x);
    auto input = Kokkos::mdspan(m_prev->output(), z, y_in, x_in);

    for (int c = 0; c < z; ++c)
        for (int i = 0; i < y; ++i) {
            int in_i = i / SCALE_FACTOR;
            for (int j = 0; j < x; ++j)
                output(c, i, j) = input(c, in_i, j / SCALE_FACTOR);
        }
}

void UpSample2DCPU::backward(float learning_rate, const float* grad_output)
{
    auto [x, y, z] = dimension();
    auto [x_in, y_in, _] = m_prev->dimension();

    std::fill(grad_input.begin(), grad_input.end(), 0);
    auto grad_input_3d = Kokkos::mdspan(grad_input.data(), z, y_in, x_in);
    auto grad_output_3d = Kokkos::mdspan(grad_output, z, y, x);

    for (int c = 0; c < z; ++c) {
        for (int i = 0; i < y; ++i) {
            int i_in = i / SCALE_FACTOR; 
            for (int j = 0; j < x; ++j)
                grad_input_3d(c, i_in, j / SCALE_FACTOR) += grad_output_3d(c, i, j);            
        }
    }

    m_prev->backward(learning_rate, grad_input.data());
}
