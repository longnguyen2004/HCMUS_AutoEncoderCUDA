#pragma once
#include "../base/layer_base.h"

class MaxPool2DCPU : public LayerCPU
{
public:
  MaxPool2DCPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float *grad_output);
  std::tuple<int, int, int> dimension() const;
};

class MaxPool2DGPU : public LayerGPU
{
public:
  MaxPool2DGPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float *grad_output);
  std::tuple<int, int, int> dimension() const;
};
