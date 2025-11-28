cmake_minimum_required(VERSION 3.10)
project(autoencoder CXX CUDA)

add_executable(
  AUTOENCODER_EXEC
    src/convolution/convolution_cpu.cpp
    src/hello.cpp
    src/stbi.cpp
    src/stbi_write.cpp
)
target_include_directories(AUTOENCODER_EXEC PRIVATE include)
target_compile_features(AUTOENCODER_EXEC PRIVATE cxx_std_20)
