#pragma once
#include "layer/base/layer_base.h"

class Conv2D : public Layer {
  private:
    Tensor weight, bias;
    int c_in, c_out;
  public:
    Conv2D(int in_channels, int out_channels);
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad_output) override;
    void step(float lr);
};
