#include "convolution.h"
#include <algorithm>

void convolve_cpu(float* dst, float* src, float* kernel, int row, int col)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      float val = 0;
      for (int k = 0; k < KERNEL_WIDTH; ++k)
      {
        for (int l = 0; l < KERNEL_WIDTH; ++l)
        {
          int i_mapped = std::clamp(i + k - KERNEL_RADIUS, 0, row - 1);
          int j_mapped = std::clamp(j + l - KERNEL_RADIUS, 0, col - 1);
          val += src[i_mapped * col + j_mapped] * kernel[k * KERNEL_WIDTH + l];
        }
      }
      dst[i * col + j] = val;
    }
  }
}

void ConvolutionCpu::convolve(
  float *dst, float * src, float* kernel, int row, int col
)
{
  convolve_cpu(dst, src, kernel, row, col);
}
