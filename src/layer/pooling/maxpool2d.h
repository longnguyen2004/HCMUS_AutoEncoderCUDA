#pragma once
#include "layer/base/layer_base.h"
#include <memory>

class MaxPool2DGPU : public Layer
{
public:
  MaxPool2DGPU(std::shared_ptr<Layer> prev, int stride_x, int stride_y);
  ~MaxPool2DGPU();
  void forward() = 0;
  void backward(float learning_rate) = 0;
  const float* output() const;
  std::tuple<int, int, int> dimension() const;
  size_t paramCount() const;
  void setParams(float* params) {};

  DeviceType deviceType() const { return DeviceType::GPU; }

private:
  std::shared_ptr<Layer> m_prev;
  float *m_output;
  int m_stride_x, m_stride_y;
};
