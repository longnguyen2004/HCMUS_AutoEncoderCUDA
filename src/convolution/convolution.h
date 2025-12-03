#include <cstdint>
#include <cuda_runtime.h>
#include <constants.h>

__host__ __device__ int inline mapIndex(int index, int bound) {
  if (index < 0 || index >= bound) return 0;
  else return index;
}

class Convolution
{
public:
  virtual void convolve(
      float *dst, float * src, float* kernel, int row, int col) = 0;
};

class ConvolutionCpu : public Convolution
{
public:
  void convolve(
    float *dst, float * src, float* kernel, int row, int col);
};

class ConvolutionGpu: public Convolution
{
public: 
  void convolve(
    float *dst, float * src, float* kernel, int row, int col);
};
