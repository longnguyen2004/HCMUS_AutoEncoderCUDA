#pragma once
constexpr int KERNEL_WIDTH = 3;
constexpr int KERNEL_RADIUS = KERNEL_WIDTH / 2;
constexpr int KERNEL_SIZE = KERNEL_WIDTH * KERNEL_WIDTH;
constexpr int BLOCK_SIZE_1D = 256;
constexpr int BLOCK_SIZE_2D = 16;
constexpr int IMAGE_DIMENSION = 32; 
constexpr int IMAGE_PIXELS = IMAGE_DIMENSION * IMAGE_DIMENSION;
constexpr size_t IMAGE_BYTE_SIZE = sizeof(float) * 3 * IMAGE_PIXELS;
constexpr int MAXPOOL2D_STRIDE = 2;