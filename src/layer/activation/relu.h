#pragma once
#include "../base/layer_base.h"
#include <memory>

class ReluGPU : public Layer {
public:
  ReluGPU(std::shared_ptr<Layer> prev);
  ~ReluGPU();
  void forward();
  void backward(float learning_rate);
  const float* output() const;
  std::tuple<int, int, int> dimension() const;
  size_t paramCount() const;
  void setParams(float* params) {};

  DeviceType deviceType() const { return DeviceType::GPU; }

private:
  std::shared_ptr<Layer> m_prev;
  float* m_output;
};
