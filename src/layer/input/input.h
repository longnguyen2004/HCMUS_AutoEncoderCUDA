#include "../base/layer_base.h"

class InputCPU : public LayerCPU {
public:
  InputCPU(const std::vector<float>& input);
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  void forward() override { return; };
  void backward(float learning_rate, const float* grad_output) override { return; };
};

class InputGPU : public LayerGPU {
public:
  InputGPU(const std::vector<float>& input);
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  void forward() override {};
  void backward(float learning_rate, const float* grad_output) override {};
}; 