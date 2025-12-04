#include "core/tensor.h"
#include "planar/planar.h"
#include "helper/gpu_helper.h"
#include <stb_image.h>
#include <stb_image_write.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <convolution/cpu/convolution_cpu.h>
#include <convolution/gpu/convolution_gpu.h>

int main(int argc, char const *argv[])
{
    int x, y, file_channels;
    // Force loading as 3 channels (RGB) to ensure consistent data layout
    auto img = stbi_load("in.png", &x, &y, &file_channels, 3);
    if (!img) {
        std::cerr << "Failed to load image in.png" << std::endl;
        return 1;
    }
    
    int channels = 3;
    std::cout << "Image loaded: " << x << "x" << y << " original channels=" << file_channels << " (forced to 3)" << std::endl;
    
    std::vector<float> in(x * y * channels);
    std::vector<float> out_cpu(x * y * channels);
    std::vector<float> out_gpu(x * y * channels);
    std::vector<float> in_r(x * y), in_g(x * y), in_b(x * y);
    std::vector<float> out_r(x * y), out_g(x * y), out_b(x * y);
    
    std::copy(img, img + (x * y * channels), in.begin());
    stbi_image_free(img);
    
    packed_to_planar(in.data(), in_r.data(), in_g.data(), in_b.data(), x * y);
    
    float n = 1.0f/9.0f;
    std::vector<float> kernel = {n,n,n,n,n,n,n,n,n};
    
    // Test CPU
    std::cout << "Testing CPU implementation..." << std::endl;
    std::unique_ptr<Convolution> convolver_cpu(new ConvolutionCpu);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    Tensor tensor({100, 3, y, x}, GPU);
    std::cout << "Tensor shape: n=" << tensor.shape.n << ", c=" << tensor.shape.c 
              << ", h=" << tensor.shape.h << ", w=" << tensor.shape.w << std::endl;
    std::cout << "Tensor size: " << tensor.shape.size() << " floats" << std::endl;
    for (int i = 0; i < 100; i++)
        Tensor::from_image(tensor, i, in_r.data(), in_g.data(), in_b.data(), x * y);
    
    tensor = tensor.to(CPU);
    std::cout << "Tensor moved to CPU." << std::endl;
    for (int row = 0; row < y; row++) {
        for (int col = 0; col < x; col++) {
            if (tensor(99, 0, row, col) != in_r[row * x + col] ||
                tensor(99, 1, row, col) != in_g[row * x + col] ||
                tensor(99, 2, row, col) != in_b[row * x + col]) {
                std::cerr << "Tensor from_image error at (" << row << ", " << col << ")" << std::endl;
                return 1;
            }
        }
    }

    convolver_cpu->convolve(out_r.data(), in_r.data(), kernel.data(), y, x);
    convolver_cpu->convolve(out_g.data(), in_g.data(), kernel.data(), y, x);
    convolver_cpu->convolve(out_b.data(), in_b.data(), kernel.data(), y, x);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;
    
    planar_to_packed(out_cpu.data(), out_r.data(), out_g.data(), out_b.data(), x * y);
    std::vector<std::uint8_t> out_cpu_u8(out_cpu.begin(), out_cpu.end());
    stbi_write_png("out_cpu.png", x, y, channels, out_cpu_u8.data(), x * channels);
    std::cout << "CPU output saved to out_cpu.png" << std::endl;
    
    // Test GPU
    std::cout << "\nTesting GPU implementation..." << std::endl;
    std::unique_ptr<Convolution> convolver_gpu(new ConvolutionGpu);
    GPUTimer timer;
    timer.Start();
    
    convolver_gpu->convolve(out_r.data(), in_r.data(), kernel.data(), y, x);
    convolver_gpu->convolve(out_g.data(), in_g.data(), kernel.data(), y, x);
    convolver_gpu->convolve(out_b.data(), in_b.data(), kernel.data(), y, x);
    
    timer.Stop();
    float gpu_time = timer.Elapsed();
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    
    planar_to_packed(out_gpu.data(), out_r.data(), out_g.data(), out_b.data(), x * y);
    std::vector<std::uint8_t> out_gpu_u8(out_gpu.begin(), out_gpu.end());
    stbi_write_png("out_gpu.png", x, y, channels, out_gpu_u8.data(), x * channels);
    std::cout << "GPU output saved to out_gpu.png" << std::endl;
    
    double diff = checkDiff(out_cpu.data(), out_gpu.data(), x * y * channels);
    std::cout << "Average Error: " << diff << std::endl;
    std::cout << "Image size: " << x << "x" << y << " (" << channels << " channels)" << std::endl;
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    if (gpu_time > 0)
        std::cout << "Speedup: " << (float)duration_cpu.count() / gpu_time << "x" << std::endl;
    
    return 0;
}
