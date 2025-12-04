#include "../maxpool2d.h"

MaxPool2DGPU::MaxPool2DGPU(std::shared_ptr<Layer> prev, int stride_x, int stride_y):
    m_prev(prev), m_stride_x(stride_x), m_stride_y(stride_y)
{
    auto [x, y, z] = dimension();
    cudaMalloc(reinterpret_cast<void**>(&m_output), x * y * z * sizeof(float));
}

MaxPool2DGPU::~MaxPool2DGPU()
{
    cudaFree(m_output);
}

const float* MaxPool2DGPU::output() const
{
    return m_output;
}

std::tuple<int, int, int> MaxPool2DGPU::dimension() const
{
    auto [x, y, z] = m_prev->dimension();
    return {
        (x + m_stride_x - 1) / m_stride_x,
        (y + m_stride_y - 1) / m_stride_y,
        z
    };
}

size_t MaxPool2DGPU::paramCount() const
{
    return 0;
}

void MaxPool2DGPU::forward() {
    // TODO: Implement max pooling forward pass
}

void MaxPool2DGPU::backward(float learning_rate) {
    // TODO: Implement max pooling backward pass
}
