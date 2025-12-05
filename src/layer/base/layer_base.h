#pragma once
#include <tuple>
#include <cstddef>

enum DeviceType { CPU, GPU };

class Layer {
public: 
  virtual ~Layer() = default;
  virtual void forward() = 0;
  virtual void backward(float learning_rate, const float* grad_output) = 0;
  virtual const float* output() const = 0;
  virtual std::tuple<int, int, int> dimension() const = 0;
  virtual size_t paramCount() const
  {
    return 0;
  }
  virtual void setParams(float* params)
  {
  };

  virtual DeviceType deviceType() const = 0;
};
