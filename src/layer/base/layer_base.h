#pragma once
#include <tuple>
#include <cstddef>
#include <memory>
#include <vector>

enum DeviceType { CPU, GPU };

class Layer {
protected:
  std::shared_ptr<Layer> m_prev;
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

class LayerCPU : public Layer{
protected:
  std::vector<float> m_output;
  std::vector<float> grad_input;
public: 
  const float* output() const override
  {
    return m_output.data();
  }
  DeviceType deviceType() const override {return DeviceType::CPU;};
};

class LayerGPU : public Layer{
protected:
  float* m_output = nullptr;
  float* grad_input = nullptr;
public: 
  virtual ~LayerGPU();
  const float* output() const override
  {
    return m_output;
  }
  DeviceType deviceType() const override {return DeviceType::GPU;};
};
