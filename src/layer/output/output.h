#pragma once
#include <layer/base/layer_base.h>
#include <memory>
#include <vector>

// Mean-squared-error output layer: compares network prediction to a reference image
// and propagates the gradient back to the previous layer.
class OutputCPU : public LayerCPU {
public:
  explicit OutputCPU(std::shared_ptr<Layer> prev);
  void setReferenceImage(const std::vector<float>& image);
  void forward() override;
  void backward(float learning_rate, const float* /*grad_output*/) override;
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  float loss() const;

private:
  const std::vector<float>* m_target = nullptr;
  float m_loss = 0.0f;
};

class OutputGPU : public LayerGPU {
public:
  explicit OutputGPU(std::shared_ptr<Layer> prev);
  ~OutputGPU() override;
  void setReferenceImage(const std::vector<float>& image);
  void forward() override;
  void backward(float learning_rate, const float* /*grad_output*/) override;
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  float loss() const;

private:
  std::vector<float> h_target;
  float* d_target = nullptr;
  float* d_loss = nullptr;
  float m_loss = 0.0f;
};
