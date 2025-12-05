#pragma once
#include "../base/layer_base.h"
#include <memory>
#include <vector>

class ReluCPU : public LayerCPU {
public:
  ReluCPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
};

class ReluGPU : public LayerGPU {
public:
  ReluGPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
};