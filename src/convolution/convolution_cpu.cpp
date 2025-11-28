#include "convolution.h"
#include <algorithm>

void ConvolutionCpu::convolve(Image dst, Image src, float *kernel, int row, int col, int kernel_size)
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
          int i_mapped = std::clamp(i + k - kernel_size / 2, 0, row);
          int j_mapped = std::clamp(j + l - kernel_size / 2, 0, col);
          r += src[i_mapped][j_mapped][0] * kernel[k * kernel_size + l];
          g += src[i_mapped][j_mapped][1] * kernel[k * kernel_size + l];
          b += src[i_mapped][j_mapped][2] * kernel[k * kernel_size + l];
        }
      }
      dst[i][j][0] = r;
      dst[i][j][1] = g;
      dst[i][j][2] = b;
    }
  }
}
