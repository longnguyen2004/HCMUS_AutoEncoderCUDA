#include "../input.h"
#include <constants.h>

InputCPU::InputCPU() {};

void InputCPU::setImage(const std::vector<float> &image) {
  m_image = &image;
}

const float* InputCPU::output() const {
  return m_image->data();
}

std::tuple<int, int, int> InputCPU::dimension() const {
  return std::make_tuple(IMAGE_DIMENSION, IMAGE_DIMENSION, 3);
}


