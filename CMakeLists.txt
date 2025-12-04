cmake_minimum_required(VERSION 3.10)
project(autoencoder CXX CUDA)

add_executable(
  AUTOENCODER_EXEC
    # Core
    src/core/tensor.cu
    
    # Layers
    src/layer/activation/relu.cu
    src/layer/conv/conv2d.cu
    src/layer/pooling/maxpool2d.cu
    src/layer/upsampling/upsample2d.cu
    
    # Convolution
    src/convolution/cpu/convolution_cpu.cpp
    src/convolution/gpu/convolution_gpu.cu
    
    # Utils
    src/hello.cpp
    src/stbi.cpp
    src/stbi_write.cpp
)
target_include_directories(AUTOENCODER_EXEC PRIVATE include src)
target_compile_features(AUTOENCODER_EXEC PRIVATE cxx_std_20)
