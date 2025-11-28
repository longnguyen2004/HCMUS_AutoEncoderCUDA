#include <cstdint>

using ImageIn = std::uint8_t[32][32][3];
using ImageOut = float[32][32][3];

class Convolution
{
  virtual void convolve(
    ImageOut dst, ImageIn src, float *kernel, int x, int y, int kernel_size
  ) = 0;
};

class ConvolutionCpu: public Convolution
{
  void convolve(
    ImageOut dst, ImageIn src, float *kernel, int x, int y, int kernel_size
  );
};
