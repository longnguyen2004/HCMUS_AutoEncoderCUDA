#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <constants.h>

__host__ __device__ inline int mapIndex(int index, int bound) {
  if (index < 0 || index >= bound) return 0;
  else return index;
}

class Convolution {
public:
  virtual ~Convolution() = default;
  virtual void convolve(
      float *dst, float *src, float *kernel, int row, int col) = 0;
};
