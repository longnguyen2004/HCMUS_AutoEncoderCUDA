#pragma once
#include <cuda_runtime.h>
#include <constants.h>
#include <mdspan/mdspan.hpp>
#include <memory>
#include <cstdint>

enum DeviceType { CPU, GPU };

using tensor_view = Kokkos::mdspan<
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
  int n, c, h, w;
  size_t size();
};

struct Tensor {
  private:
    std::shared_ptr<float[]> buffer;
    Tensor(std::shared_ptr<float[]> existing_buf, Shape s, DeviceType d);

  public:
    tensor_view view;
    Shape shape; 
    DeviceType device;
    
    Tensor(Shape shape, DeviceType device = CPU);
    Tensor to(DeviceType target_device);
    Tensor reshape(Shape new_shape);
    DeviceType get_device();
    void randomize(); //to randomly initialized weights
    float& operator()(size_t n, size_t c, size_t h, size_t w);
    float* data();
    const float* data() const;
    static void from_image(Tensor& tensor, size_t index, const float* r, const float* g, const float* b, const size_t pixels = IMAGE_PIXELS);
};