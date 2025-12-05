#include "../upsample2d.h"
#include <constants.h>

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
    return;
}

void UpSample2DCPU::backward(float learning_rate, const float* grad_output)
{
    return;
}
