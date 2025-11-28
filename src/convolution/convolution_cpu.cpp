#include "convolution.h"
#include <algorithm>

void ConvolutionCpu::convolve(
  float *dst_r, float *dst_g, float *dst_b,
  float *src_r, float *src_g, float *src_b,
  float *kernel, int row, int col, int kernel_size)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      float r = 0, g = 0, b = 0;
      for (int k = 0; k < kernel_size; ++k)
      {
        for (int l = 0; l < kernel_size; ++l)
        {
          int i_mapped = std::clamp(i + k - kernel_size / 2, 0, row - 1);
          int j_mapped = std::clamp(j + l - kernel_size / 2, 0, col - 1);
          r += src_r[i_mapped * col + j_mapped] * kernel[k * kernel_size + l];
          g += src_g[i_mapped * col + j_mapped] * kernel[k * kernel_size + l];
          b += src_b[i_mapped * col + j_mapped] * kernel[k * kernel_size + l];
        }
      }
      dst_r[i * col + j] = r;
      dst_g[i * col + j] = g;
      dst_b[i * col + j] = b;
    }
  }
}
