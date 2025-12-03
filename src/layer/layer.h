#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <constants.h>
#include <mdspan/mdspan.hpp>
#include <memory>
#include <cstring>

enum DeviceType { CPU, GPU };

using tensor_mdspan = Kokkos::mdspan<
    float,
    Kokkos::extents<size_t, Kokkos::dynamic_extent,
                         Kokkos::dynamic_extent,
                         Kokkos::dynamic_extent,
                         Kokkos::dynamic_extent>
>;

struct Shape {
  //N : number of inputs
  //C : number of channels
  //H : height of image
  //W : width of image
  unsigned int n, c, h, w;
  size_t size();
};

struct Tensor {
  private:
    Tensor(std::shared_ptr<float[]> existing_buf, Shape s, DeviceType d);

  public:
    std::shared_ptr<float[]> buffer;
    tensor_mdspan view;
    Shape shape; 
    DeviceType device;
    Tensor(Shape shape, DeviceType device = CPU);
    Tensor to(DeviceType target_device);
    Tensor reshape(Shape new_shape);
    DeviceType get_device();
    void randomize(); //to randomly initialized weights
    float& operator()(uint32_t n, uint32_t c, uint32_t h, uint32_t w);
    
    static void from_image(Tensor& tensor, int index, const float* r, const float* g, const float* b);
};

class Layer {
  protected: 
    Tensor* cached_input = nullptr; //for back-propagation
  public: 
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor& input) = 0;
    virtual Tensor backward(Tensor& grad_output) = 0;
    virtual void step(float lr) = 0;
};

class Conv2D : public Layer {
  private:
    Tensor weight, bias;
    int c_in, c_out;
  public:
    Conv2D(int in_channels, int out_channels);
};

class Relu : public Layer {};
class MaxPool2D : public Layer {};
class UpSample2D : public Layer {};