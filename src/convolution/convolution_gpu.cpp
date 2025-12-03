#include "convolution.h"
#include "helper/GPU_helper.h"

__global__ void convolve_gpu(float* dst, float* src, float* kernel, int row, int col, int kernel_size)  
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= row || x >= col) return;
    float val = 0;
    for (int yf = 0; yf < kernel_size; ++yf) {
        for (int xf = 0; xf < kernel_size; ++xf) {
            int y_mapped = mapIndex(y + yf - kernel_size / 2, row);
            int x_mapped = mapIndex(x + xf - kernel_size / 2, col);
            val += src[y_mapped * col + x_mapped] * kernel[yf * kernel_size + xf]; 
        }
    }
    dst[y * col + x] = val;
}

__constant__ float kernel[3 * 3];

void ConvolutionGpu::convolve(
  float *dst_r, float *dst_g, float *dst_b,
  float *src_r, float *src_g, float *src_b,
  float *kernel_r, float *kernel_g, float *kernel_b,
  int row, int col, int kernel_size, dim3 blockSize) 
{
    float *cudaInPixels, *cudaOutPixels;
    CHECK(cudaMalloc((void**)&cudaInPixels, sizeof(float) * row * col));
    CHECK(cudaMalloc((void**)&cudaOutPixels, sizeof(float) * row * col));

    dim3 gridSize(
        (col + blockSize.x - 1) / blockSize.x,
        (row + blockSize.y - 1) / blockSize.y
    );

    //r
    CHECK(cudaMemcpy(cudaInPixels, src_r, sizeof(float) * row * col, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpyToSymbol(kernel, kernel_r, sizeof(float) * 3 * 3, 0, cudaMemcpyHostToDevice));
    convolve_gpu<<<gridSize, blockSize>>>(cudaOutPixels, cudaInPixels, kernel, row, col, kernel_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(dst_r, cudaOutPixels, sizeof(float) * row * col, cudaMemcpyDeviceToHost));
    //g 
    CHECK(cudaMemcpy(cudaInPixels, src_g, sizeof(float) * row * col, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpyToSymbol(kernel, kernel_g, sizeof(float) * 3 * 3, 0, cudaMemcpyHostToDevice));
    convolve_gpu<<<gridSize, blockSize>>>(cudaOutPixels, cudaInPixels, kernel, row, col, kernel_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(dst_g, cudaOutPixels, sizeof(float) * row * col, cudaMemcpyDeviceToHost));
    //b
    CHECK(cudaMemcpy(cudaInPixels, src_b, sizeof(float) * row * col, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpyToSymbol(kernel, kernel_b, sizeof(float) * 3 * 3, 0, cudaMemcpyHostToDevice));
    convolve_gpu<<<gridSize, blockSize>>>(cudaOutPixels, cudaInPixels, kernel, row, col, kernel_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(dst_b, cudaOutPixels, sizeof(float) * row * col, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(cudaInPixels));
    CHECK(cudaFree(cudaOutPixels));
}