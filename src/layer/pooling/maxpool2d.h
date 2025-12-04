#pragma once
#include "layer/base/layer_base.h"

class MaxPool2D : public Layer {
  private:
    int pool_size;
    int stride;
  public:
    MaxPool2D(int pool_size, int stride);
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad_output) override;
};
