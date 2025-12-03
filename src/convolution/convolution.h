#include <cstdint>
#include <cuda_runtime.h>

__host__ __device__ int inline mapIndex(int index, int bound) {
  if (index < 0 || index >= bound) return 0;
  else return index;
}

class Convolution
{
public:
  virtual void convolve(
      float *dst_r, float *dst_g, float *dst_b,
      float *src_r, float *src_g, float *src_b,
      float *kernel_r, float *kernel_g, float *kernel_b,
      int row, int col, int kernel_size) = 0;
};

class ConvolutionCpu : public Convolution
{
public:
  void convolve(
      float *dst_r, float *dst_g, float *dst_b,
      float *src_r, float *src_g, float *src_b,
      float *kernel_r, float *kernel_g, float *kernel_b,
      int row, int col, int kernel_size);
};

class ConvolutionGpu: public Convolution
{
public: 
  void convolve(
      float *dst_r, float *dst_g, float *dst_b,
      float *src_r, float *src_g, float *src_b,
      float *kernel_r, float *kernel_g, float *kernel_b,
      int row, int col, int kernel_size, dim3 blockSize = dim3(1, 1));
};
