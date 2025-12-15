cmake_minimum_required(VERSION 3.10)
project(autoencoder CXX CUDA)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # enable cuda-gdb
endif()

include(FetchContent)
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan
  GIT_TAG        537053f83150f1b6f7528c1d961866d8628abc2e
)
FetchContent_MakeAvailable(mdspan)

add_executable(
  AUTOENCODER_EXEC
    # Layers
    src/layer/input/cpu/input.cpp
    src/layer/input/gpu/input.cu
    src/layer/activation/cpu/relu.cpp
    src/layer/activation/gpu/relu.cu
    src/layer/conv/cpu/conv2d.cpp
    src/layer/conv/gpu/conv2d.cu
    src/layer/pooling/gpu/maxpool2d.cu
    src/layer/pooling/cpu/maxpool2d.cpp
    src/layer/upsampling/cpu/upsample2d.cpp
    src/layer/upsampling/gpu/upsample2d.cu
    src/layer/output/cpu/output.cpp
    src/layer/output/gpu/output.cu

    # Core
    src/core/optimizer.cpp
    src/core/optimizer.cu
    src/core/network.cpp
    src/core/network.cu

    # SVM
    src/svm/svm.cpp

    # Utils
    src/main.cpp
    src/stbi.cpp
    src/stbi_write.cpp
)
set(CMAKE_CUDA_ARCHITECTURES 75)
target_link_libraries(AUTOENCODER_EXEC PRIVATE mdspan)
target_include_directories(AUTOENCODER_EXEC PRIVATE include src)
target_compile_features(AUTOENCODER_EXEC PRIVATE cxx_std_20)

# Evaluation executable
add_executable(
  AUTOENCODER_EVAL
    # Layers
    src/layer/input/cpu/input.cpp
    src/layer/input/gpu/input.cu
    src/layer/activation/cpu/relu.cpp
    src/layer/activation/gpu/relu.cu
    src/layer/conv/cpu/conv2d.cpp
    src/layer/conv/gpu/conv2d.cu
    src/layer/pooling/gpu/maxpool2d.cu
    src/layer/pooling/cpu/maxpool2d.cpp
    src/layer/upsampling/cpu/upsample2d.cpp
    src/layer/upsampling/gpu/upsample2d.cu
    src/layer/output/cpu/output.cpp
    src/layer/output/gpu/output.cu

    # Core
    src/core/optimizer.cpp
    src/core/optimizer.cu

    # Utils
    src/evaluate.cpp
    src/stbi.cpp
    src/stbi_write.cpp
)
target_link_libraries(AUTOENCODER_EVAL PRIVATE mdspan)
target_include_directories(AUTOENCODER_EVAL PRIVATE include src)
target_compile_features(AUTOENCODER_EVAL PRIVATE cxx_std_20)
