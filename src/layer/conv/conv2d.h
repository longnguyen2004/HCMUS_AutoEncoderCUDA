#pragma once
#include <layer/base/layer_base.h>
#include <memory>
#include <vector>

class Conv2DCPU : public Layer {
public:
  Conv2DCPU(std::shared_ptr<Layer> prev, int kernel_size, int filters);
  ~Conv2DCPU() = default;
  void forward();
  void backward(float learning_rate, const float* grad_output);
  const float* output() const;
  std::tuple<int, int, int> dimension() const;
  size_t paramsCount() const;
  void setParams(float* params);

  DeviceType deviceType() { return DeviceType::CPU; }
private:
  std::shared_ptr<Layer> m_prev;
  std::vector<float> m_output;
  float *m_weights, *m_biases;
  int m_kernel_size, m_filters;
};

class Conv2DGPU : public Layer {
public:
  Conv2DGPU(std::shared_ptr<Layer> prev, int kernel_size, int filters);
  ~Conv2DGPU();
  void forward();
  void backward(float learning_rate, const float* grad_output);
  const float* output() const;
  std::tuple<int, int, int> dimension() const;
  size_t paramsCount() const;
  void setParams(float* params);

  DeviceType deviceType() { return DeviceType::CPU; }
private:
  std::shared_ptr<Layer> m_prev;
  float *m_output, *m_weights, *m_biases;
  int m_kernel_size, m_filters;
};
