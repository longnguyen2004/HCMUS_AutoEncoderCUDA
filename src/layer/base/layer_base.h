#pragma once
#include <tuple>

enum DeviceType { CPU, GPU };

class Layer {
public: 
  virtual ~Layer() = default;
  virtual void forward() = 0;
  virtual void backward(float learning_rate) = 0;
  virtual const float* output() const = 0;
  virtual std::tuple<int, int, int> dimension() const = 0;
  virtual size_t paramCount() const = 0;
  virtual void setParams(float* params) = 0;

  virtual DeviceType deviceType() const = 0;
};
