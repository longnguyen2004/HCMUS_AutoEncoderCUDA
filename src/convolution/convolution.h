#include <cstdint>

class Convolution
{
public:
  virtual void convolve(
    float* dst_r, float* dst_g, float* dst_b,
    float* src_r, float* src_g, float* src_b,
    float *kernel, int row, int col, int kernel_size
  ) = 0;
};

class ConvolutionCpu: public Convolution
{
public:
  void convolve(
    float* dst_r, float* dst_g, float* dst_b,
    float* src_r, float* src_g, float* src_b,
    float *kernel, int row, int col, int kernel_size
  );
};
