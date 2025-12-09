#pragma once
#include <layer/base/layer_base.h>

class Conv2DCPU : public LayerCPU {
public:
  Conv2DCPU(std::shared_ptr<Layer> prev, int kernel_size, int filters);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
  size_t paramCount() const override;
  void setParams(float* params) override;

private:
  std::vector<float> m_grad_weights, m_grad_biases;
  float *m_weights, *m_biases;
  int m_kernel_size, m_filters;
};

class Conv2DGPU : public LayerGPU {
public:
  Conv2DGPU(std::shared_ptr<Layer> prev, int kernel_size, int filters);
  void forward();
  void backward(float learning_rate, const float* grad_output);
  std::tuple<int, int, int> dimension() const;
  size_t paramCount() const override;
  void setParams(float* params) override;

private:
  float *m_weights, *m_biases, *grad_weights, *grad_biases;
  int m_kernel_size, m_filters;
};
