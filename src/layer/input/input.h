#include "../base/layer_base.h"

class InputCPU : public LayerCPU {
public:
  InputCPU();
  void setImage(const std::vector<float> &image);
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  void forward() override { return; };
  void backward(float learning_rate, const float* grad_output) override {};
private:
  const std::vector<float> *m_image;
};

class InputGPU : public LayerGPU {
public:
  InputGPU();
  void setImage(const std::vector<float> &image);
  std::tuple<int, int, int> dimension() const override;
  const float* output() const override;
  void forward() override {};
  void backward(float learning_rate, const float* grad_output) override {};
}; 