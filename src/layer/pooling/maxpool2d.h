#pragma once
#include "../base/layer_base.h"
#include <memory>
#include <vector>

class MaxPool2DCPU : public Layer
{
private:
  int m_stride;
  std::shared_ptr<Layer> m_prev;
  std::vector<float> m_output;

public:
  MaxPool2DCPU(std::shared_ptr<Layer> prev, int stride);
  void forward();
  void backward(float learning_rate, const float *grad_output);
  const float *output() const;
  std::tuple<int, int, int> dimension() const;

  DeviceType deviceType() const { return DeviceType::CPU; }
};

class MaxPool2DGPU : public Layer
{
private:
  int m_stride;
  std::shared_ptr<Layer> m_prev;
  float* m_output;

public:
  MaxPool2DGPU(std::shared_ptr<Layer> prev, int stride);
  void forward();
  void backward(float learning_rate, const float *grad_output);
  const float *output() const;
  std::tuple<int, int, int> dimension() const;

  DeviceType deviceType() const { return DeviceType::GPU; }
};
