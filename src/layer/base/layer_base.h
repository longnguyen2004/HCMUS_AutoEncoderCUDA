#pragma once
#include "core/tensor.h"

class Layer {
  protected: 
    Tensor* cached_input = nullptr; //for back-propagation
  public: 
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor& input) = 0;
    virtual Tensor backward(Tensor& grad_output) = 0;
};
