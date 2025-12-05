#pragma once
#include <layer/base/layer_base.h>

class UpSample2D : public Layer {
  private:
    int scale_factor;
  public:
    UpSample2D(int scale);
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad_output) override;
};
