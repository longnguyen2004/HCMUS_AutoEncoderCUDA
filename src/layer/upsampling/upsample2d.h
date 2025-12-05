#pragma once
#include <layer/base/layer_base.h>

class UpSample2DCPU : public LayerCPU {
public:
  UpSample2DCPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
};

class UpSample2DGPU : public LayerGPU {
public:
  UpSample2DGPU(std::shared_ptr<Layer> prev);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
};
