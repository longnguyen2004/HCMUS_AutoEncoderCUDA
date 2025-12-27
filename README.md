# Setting up the project

## Dependencies

- **The NVIDIA CUDA SDK** (Target architecture: 75)
- **A C++20 compliant compiler** (e.g., GCC 10+, Clang 10+, or MSVC 2019+)
- **CMake 3.10+**
- **mdspan** (Automatically handled by CMake via FetchContent)

## Conv2D Implementation Options

The project includes 3 different Conv2D GPU implementations in `src/layer/conv/gpu/conv2d.cu`:

1. **Naive** (`IMPLEMENTATION=1`): Simple direct convolution kernel without shared memory.
2. **Optimized Naive** (`IMPLEMENTATION=2`): Tiled convolution with shared memory and tree reduction for gradient weights.
3. **Im2Col** (`IMPLEMENTATION=3`): Matrix multiplication using the im2col transformation (Default).

Switch between implementations by changing the `#define IMPLEMENTATION` at the top of `src/layer/conv/gpu/conv2d.cu`.

## Compilation command

1. In the project root, run:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   ```
2. Build the project:
   ```bash
   cmake --build build -j
   ```
   *Note: On Windows with Visual Studio, use `cmake --build build --config Release`.*

## Executing the code

Assuming `path/to/cifar-10-batches-bin` contains the CIFAR-10 dataset (`data_batch_n.bin` and `test_batch.bin`).

### Training AutoEncoder

```bash
./build/AUTOENCODER_EXEC [path/to/dataset]      # GPU version
./build/AUTOENCODER_EXEC_CPU [path/to/dataset]  # CPU version
```

Weights will be output to `params_epoch_n.bin` in the current working directory after each epoch.

### Evaluating AutoEncoder

```bash
./build/AUTOENCODER_EVAL [path/to/dataset]      # GPU version
./build/AUTOENCODER_EVAL_CPU [path/to/dataset]  # CPU version
```

A folder `test_outputs` will be created, containing original and reconstructed versions of the first 20 images in the dataset.

### Extract Features for SVM (GPU only)

```bash
./build/EXTRACT_FEATURES [path/to/dataset]
```

Four CSV files will be created: `train_features.csv`, `train_labels.csv`, `test_features.csv`, and `test_labels.csv`, containing training and testing data for SVM.
