#include "../output.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <utility>

OutputCPU::OutputCPU(std::shared_ptr<Layer> prev) {
  m_prev = std::move(prev);
  auto [x, y, z] = dimension();
  grad_input.resize(static_cast<size_t>(x) * y * z);
}

std::tuple<int, int, int> OutputCPU::dimension() const {
  return m_prev->dimension();
}

const float* OutputCPU::output() const {
  return m_prev->output();
}

float OutputCPU::loss() const {
  return m_loss;
}

void OutputCPU::setReferenceImage(const std::vector<float>& image) {
  auto [x, y, z] = dimension();
  const size_t expected = static_cast<size_t>(x) * y * z;
  if (image.size() != expected) {
    throw std::runtime_error("OutputCPU: reference image size does not match network output");
  }
  m_target = &image;
}

void OutputCPU::forward() {
  if (!m_target) {
    throw std::runtime_error("OutputCPU: reference image not set");
  }
  m_prev->forward();
  auto [x, y, z] = dimension();
  const size_t n = static_cast<size_t>(x) * y * z;
  const float* pred = m_prev->output();

  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double diff = static_cast<double>(pred[i]) - static_cast<double>((*m_target)[i]);
    sum += diff * diff;
  }
  m_loss = static_cast<float>(sum / static_cast<double>(n));
}

void OutputCPU::backward(float learning_rate, const float* /*grad_output*/) {
  if (!m_target) {
    throw std::runtime_error("OutputCPU: reference image not set");
  }
  auto [x, y, z] = dimension();
  const size_t n = static_cast<size_t>(x) * y * z;
  const float scale = 2.0f / static_cast<float>(n);
  const float* pred = m_prev->output();

  for (size_t i = 0; i < n; ++i) {
    grad_input[i] = scale * (pred[i] - (*m_target)[i]);
  }

  m_prev->backward(learning_rate, grad_input.data());
}
