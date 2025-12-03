#include "convolution.h"
#include "helper/gpu_helper.h"

__global__ void convolve_gpu(float* dst, float* src, float* kernel, int row, int col)  
{
    extern __shared__ float s_src[];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int blockStart_x = blockDim.x * blockIdx.x;
    int blockStart_y = blockDim.y * blockIdx.y;
    int x = blockStart_x + tx;
    int y = blockStart_y + ty;
    
    int tileWidth = blockDim.x + KERNEL_RADIUS * 2;
    int tileHeight = blockDim.y + KERNEL_RADIUS * 2;
    for (int j = ty; j < tileHeight; j += blockDim.y) {
        for (int i = tx; i < tileWidth; i += blockDim.x) {
            int gy = blockStart_y - KERNEL_RADIUS + j;
            int gx = blockStart_x - KERNEL_RADIUS + i;
            if (gy < 0 || gy >= row || gx < 0 || gx >= col)
                s_src[j * tileWidth + i] = 0;
            else 
                s_src[j * tileWidth + i] = src[gy * col + gx];
        }
    }

    __syncthreads();

    if (x >= col || y >= row) return;
    float pixel = 0.0f;
    for (int yf = 0; yf < KERNEL_WIDTH; ++yf) {
        for (int xf = 0; xf < KERNEL_WIDTH; ++xf) {
            int x_mapped = tx + xf;
            int y_mapped = ty + yf;
            pixel += s_src[tileWidth * y_mapped + x_mapped] * kernel[yf * KERNEL_WIDTH + xf];
        }
    }
    dst[col * y + x] = pixel;
}

void ConvolutionGpu::convolve(
  float *dst, float * src, float* kernel, int row, int col)
{
    float *cuda_dst, *cuda_src, *cuda_kernel;
    size_t bytes = sizeof(float) * row * col;
    CHECK(cudaMalloc((void**)&cuda_dst, bytes));
    CHECK(cudaMalloc((void**)&cuda_src, bytes));
    CHECK(cudaMalloc((void**)&cuda_kernel, sizeof(float) * KERNEL_SIZE));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (col + blockSize.x - 1) / blockSize.x,
        (row + blockSize.y - 1) / blockSize.y
    );

    size_t shared_memory = sizeof(float) * (2 * KERNEL_RADIUS + blockSize.x) * (2 * KERNEL_RADIUS + blockSize.y);
    CHECK(cudaMemcpy(cuda_src, src, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(cuda_kernel, kernel, sizeof(float) * KERNEL_SIZE, cudaMemcpyHostToDevice));
    convolve_gpu<<<gridSize, blockSize, shared_memory>>>(cuda_dst, cuda_src, cuda_kernel, row, col);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(dst, cuda_dst, bytes, cudaMemcpyDeviceToHost));

    cudaFree(cuda_dst);
    cudaFree(cuda_src);
    cudaFree(cuda_kernel);
}