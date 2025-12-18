set(CMAKE_CUDA_ARCHITECTURES 75)

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

add_library(
  AUTOENCODER_CORE
    # Layers
    src/layer/base/layer_base.cu
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

    # Utils
    src/stbi.cpp
    src/stbi_write.cpp
)
target_link_libraries(AUTOENCODER_CORE PRIVATE mdspan)
target_compile_features(AUTOENCODER_CORE PRIVATE cxx_std_20)
target_include_directories(AUTOENCODER_CORE PUBLIC include src)

add_executable(
  AUTOENCODER_EXEC
    src/main.cu
)
target_link_libraries(AUTOENCODER_EXEC PRIVATE AUTOENCODER_CORE)
target_compile_features(AUTOENCODER_EXEC PRIVATE cxx_std_20)

# Training executable (CPU version)
add_executable(
  AUTOENCODER_EXEC_CPU
    src/main_cpu.cpp
)
target_link_libraries(AUTOENCODER_EXEC_CPU PRIVATE AUTOENCODER_CORE)
target_compile_features(AUTOENCODER_EXEC_CPU PRIVATE cxx_std_20)

# Evaluation executable (GPU version)
add_executable(
  AUTOENCODER_EVAL
    src/evaluate.cu
)
target_link_libraries(AUTOENCODER_EVAL PRIVATE AUTOENCODER_CORE)
target_compile_features(AUTOENCODER_EVAL PRIVATE cxx_std_20)

# Evaluation executable (CPU version)
add_executable(
  AUTOENCODER_EVAL_CPU
    src/evaluate_cpu.cpp
)
target_link_libraries(AUTOENCODER_EVAL_CPU PRIVATE AUTOENCODER_CORE)
target_compile_features(AUTOENCODER_EVAL_CPU PRIVATE cxx_std_20)

# Training feature extraction (GPU version)
add_executable(
  EXTRACT_TRAIN
    src/extract_train.cu
)
target_link_libraries(EXTRACT_TRAIN PRIVATE AUTOENCODER_CORE)
target_compile_features(EXTRACT_TRAIN PRIVATE cxx_std_20)
