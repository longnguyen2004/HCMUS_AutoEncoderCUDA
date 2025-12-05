#include "../relu.h"
#include <cmath>

ReluCPU::ReluCPU(std::shared_ptr<Layer> prev) : m_prev(prev) {
    auto [x, y, z] = this->dimension();
    m_output.resize(x * y * z);
}

const float* ReluCPU::output() const {
    return m_output.data();
}

std::tuple<int, int, int> ReluCPU::dimension() const {
    return m_prev->dimension();
}

void ReluCPU::forward() {
    m_prev->forward();
    auto [x, y, z] = this->dimension();
    auto input = m_prev->output();
    for (int i = 0; i < x * y * z; ++i)
        m_output[i] = std::fmaxf(0.0f, input[i]);
}

void ReluCPU::backward(float learning_rate, const float* grad_output) {
    auto [x, y, z] = this->dimension();
    std::vector<float> grad_input(x * y * z);
    auto input = m_prev->output();
    for (int i = 0; i < x * y * z; ++i)
        grad_input[i] = input[i] > 0 ? grad_output[i] : 0.0f;
    this->m_prev->backward(learning_rate, grad_input.data());
}