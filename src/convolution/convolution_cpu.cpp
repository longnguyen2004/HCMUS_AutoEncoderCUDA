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
          int i_mapped = i + k - KERNEL_RADIUS;
          int j_mapped = j + l - KERNEL_RADIUS;
          
          if (i_mapped >= 0 && i_mapped < row && j_mapped >= 0 && j_mapped < col) {
             val += src[i_mapped * col + j_mapped] * kernel[k * KERNEL_WIDTH + l];
          }
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
