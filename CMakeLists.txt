cmake_minimum_required(VERSION 3.10)
project(autoencoder CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan
  GIT_TAG        537053f83150f1b6f7528c1d961866d8628abc2e
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mdspan)

add_executable(
  AUTOENCODER_EXEC
    # Core
    src/core/tensor.cu
    
    # Layers
    src/layer/activation/gpu/relu.cu
    src/layer/conv/cpu/conv2d.cpp
    src/layer/conv/gpu/conv2d.cu
    src/layer/pooling/gpu/maxpool2d.cu
    src/layer/upsampling/upsample2d.cu
    
    # Utils
    src/main.cpp
    src/stbi.cpp
    src/stbi_write.cpp
)
target_link_libraries(AUTOENCODER_EXEC PRIVATE mdspan)
target_include_directories(AUTOENCODER_EXEC PRIVATE include src)
target_compile_features(AUTOENCODER_EXEC PRIVATE cxx_std_20)
