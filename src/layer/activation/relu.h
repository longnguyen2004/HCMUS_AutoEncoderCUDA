#pragma once
#include "layer/base/layer_base.h"

class Relu : public Layer {
  public:
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad_output) override;
};
