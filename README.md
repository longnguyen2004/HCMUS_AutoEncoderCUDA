# AutoEncoder CUDA

## Hardware Requirements
- **GPU**: NVIDIA CUDA-capable GPU (tested on P100, T4, 1660 Ti)
- **Memory**: ~2GB VRAM recommended
- Any modern CUDA device with recent SM architecture will work

## Setup Instructions

### Dependencies
- The NVIDIA CUDA SDK
- A recent compiler (latest version of MSVC, GCC or Clang)
- [CMake](https://cmake.org/download/)
- Dataset: [CIFAR-10 binary](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)

### Conv2D Implementation Options
The project includes 3 different Conv2D GPU implementations in `src/layer/conv/gpu/conv2d.cu`:

1. **Naive** (`IMPLEMENTATION=1`): Simple direct convolution kernel
2. **Tiled** (`IMPLEMENTATION=2`): Optimized tiled convolution with shared memory
3. **Im2Col** (`IMPLEMENTATION=3`): Matrix multiplication using im2col transformation

Switch between implementations by changing the preprocessor define at the top of `conv2d.cu`.

## Compilation Commands
- In the project root, run `cmake -B build -DCMAKE_BUILD_TYPE=Release`
- `cd build`, then `cmake --build .`
- Note: If running on Windows and using the Visual Studio generator, run `cmake --build . --config Release` instead

## Execution Instructions

Assuming `path/to/cifar-10-batches-bin` contains the dataset (`data_batch_n.bin` and `test_batch.bin`)

### Training autoencoder
```bash
./build/AUTOENCODER_EXEC path/to/cifar-10-batches-bin      # GPU version
./build/AUTOENCODER_EXEC_CPU path/to/cifar-10-batches-bin  # CPU version
```

### Evaluating autoencoder
```bash
./build/AUTOENCODER_EVAL path/to/cifar-10-batches-bin      # GPU version
./build/AUTOENCODER_EVAL_CPU path/to/cifar-10-batches-bin  # CPU version
```

### Extract features for SVM (GPU only)
```bash
./build/EXTRACT_FEATURES path/to/cifar-10-batches-bin
```

## Expected Outputs
- **Training**: Weights will be output to `params_epoch_n.bin` in the current working directory after each epoch. Example console output:
  ```
  Batch 1 size: 10000
  Batch 2 size: 10000
  Batch 3 size: 10000
  Batch 4 size: 10000
  Batch 5 size: 10000
  Total images: 50000
  Encoder output dimension: 8 8 128
  Decoder output dimension: 32 32 3
  Initializing parameters...
  === Epoch 0 ===
  Epoch 0 Image 100 Avg Loss: 0.12258 Time: 0.269701s
  ```
- **Evaluation**: A folder `test_outputs` will be created, containing original and reconstructed versions of the first 20 images in the dataset
- **Feature Extraction**: Four CSV files will be created: `train_features.csv`, `train_labels.csv`, `test_features.csv`, `test_labels.csv` containing training and testing data for SVM

