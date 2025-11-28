#include <cstdint>

using Image = float[32][32][3];

class Convolution
{
  virtual void convolve(
    Image dst, Image src, float *kernel, int x, int y, int kernel_size
  ) = 0;
};

class ConvolutionCpu: public Convolution
{
  void convolve(
    Image dst, Image src, float *kernel, int x, int y, int kernel_size
  );
};
