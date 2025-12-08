#include "../input.h"
#include <constants.h>

InputGPU::InputGPU(const std::vector<float>& input) {
  cudaMalloc((void**)&m_output, IMAGE_BYTE_SIZE);
  cudaMemcpy(m_output, input.data(), IMAGE_BYTE_SIZE, cudaMemcpyHostToDevice);
}

std::tuple<int, int, int> InputGPU::dimension() const {
  return std::make_tuple(IMAGE_DIMENSION, IMAGE_DIMENSION, IMAGE_DIMENSION);
}

const float* InputGPU::output() const {
  return m_output;
}


