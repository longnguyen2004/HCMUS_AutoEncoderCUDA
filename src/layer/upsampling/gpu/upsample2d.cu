#include "../upsample2d.h"
#include <constants.h>

UpSample2DGPU::UpSample2DGPU(std::shared_ptr<Layer> prev)
{
    m_prev = prev;
    auto [x, y, z] = this->dimension();
    auto [in_x, in_y, in_z] = m_prev->dimension();
    cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&grad_input), in_x * in_y * in_z * sizeof(float));
}

std::tuple<int, int, int> UpSample2DGPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {x * SCALE_FACTOR, y * SCALE_FACTOR, z};
}

void UpSample2DGPU::forward()
{
    // TODO: Implement upsampling forward pass
    return;
}

void UpSample2DGPU::backward(float learning_rate, const float* grad_output)
{
    // TODO: Implement upsampling backward pass
    return;
}
