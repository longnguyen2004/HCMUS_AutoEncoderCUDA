#include "../input.h"
#include <constants.h>

InputCPU::InputCPU(const std::vector<float>& input) {
  m_output = input;
}

const float* InputCPU::output() const {
  return m_output.data();
}

std::tuple<int, int, int> InputCPU::dimension() const {
  return std::make_tuple(IMAGE_DIMENSION, IMAGE_DIMENSION, IMAGE_DIMENSION);
}


