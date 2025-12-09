#include "../input.h"
#include <constants.h>

InputGPU::InputGPU() {
  cudaMalloc((void**)&m_output, IMAGE_BYTE_SIZE);
}

void InputGPU::setImage(const std::vector<float> &image) {
  cudaMemcpy(m_output, image.data(), image.size() * sizeof(float), cudaMemcpyHostToDevice);
}

std::tuple<int, int, int> InputGPU::dimension() const {
  return std::make_tuple(IMAGE_DIMENSION, IMAGE_DIMENSION, 3);
}

const float* InputGPU::output() const {
  return m_output;
}


