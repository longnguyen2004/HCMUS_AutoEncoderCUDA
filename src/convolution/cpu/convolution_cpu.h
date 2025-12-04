#pragma once
#include "convolution/base/convolution_base.h"

class ConvolutionCpu : public Convolution {
public:
  void convolve(
    float *dst, float *src, float *kernel, int row, int col) override;
};
