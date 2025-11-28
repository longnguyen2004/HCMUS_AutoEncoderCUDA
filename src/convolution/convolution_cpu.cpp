#include "convolution.h"
#include <algorithm>

void convolve_cpu(float* dst, float* src, float* kernel, int row, int col, int kernel_size)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      float val = 0;
      for (int k = 0; k < kernel_size; ++k)
      {
        for (int l = 0; l < kernel_size; ++l)
        {
          int i_mapped = std::clamp(i + k - kernel_size / 2, 0, row - 1);
          int j_mapped = std::clamp(j + l - kernel_size / 2, 0, col - 1);
          val += src[i_mapped * col + j_mapped] * kernel[k * kernel_size + l];
        }
      }
      dst[i * col + j] = val;
    }
  }
}

void ConvolutionCpu::convolve(
  float *dst_r, float *dst_g, float *dst_b,
  float *src_r, float *src_g, float *src_b,
  float *kernel, int row, int col, int kernel_size)
{
  convolve_cpu(dst_r, src_r, kernel, row, col, kernel_size);
  convolve_cpu(dst_g, src_g, kernel, row, col, kernel_size);
  convolve_cpu(dst_b, src_b, kernel, row, col, kernel_size);
}
