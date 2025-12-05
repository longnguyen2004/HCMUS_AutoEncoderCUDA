#pragma once
#include "../base/layer_base.h"
#include <memory>
#include <vector>

class ReluGPU : public Layer {
public:
  ReluGPU(std::shared_ptr<Layer> prev);
  ~ReluGPU();
  void forward();
  void backward(float learning_rate, const float* grad_output);
  const float* output() const;
  std::tuple<int, int, int> dimension() const;

  DeviceType deviceType() const { return DeviceType::GPU; }

private:
  std::shared_ptr<Layer> m_prev;
  float* m_output;
};

class ReluCPU : public Layer {
public:
  ReluCPU(std::shared_ptr<Layer> prev);
  ~ReluCPU() = default;
  void forward();
  void backward(float learning_rate, const float* grad_output);
  const float* output() const;
  std::tuple<int, int, int> dimension() const;
  DeviceType deviceType() const { return DeviceType::CPU; }
private:
  std::shared_ptr<Layer> m_prev;
  std::vector<float> m_output;
};
