# Setting up the project

## Dependencies

- The NVIDIA CUDA SDK
- A recent compiler (latest version of MSVC, GCC or Clang)
- [CMake](https://cmake.org/download/)

## Conv2D Implementation Options

The project includes 3 different Conv2D GPU implementations in `src/layer/conv/gpu/conv2d.cu`:

1. **Naive** (`IMPLEMENTATION=1`): Simple direct convolution kernel
2. **Tiled** (`IMPLEMENTATION=2`): Optimized tiled convolution with shared memory
3. **Im2Col** (`IMPLEMENTATION=3`): Matrix multiplication using im2col transformation

Switch between implementations by changing the preprocessor define at the top of `conv2d.cu`.

## Compilation command

- In the project root, run `cmake -B build -DCMAKE_BUILD_TYPE=Release`
- `cd build`, then `cmake --build .`
- Note: If running on Windows and using the Visual Studio generator, run `cmake --build . --config Release` instead

## Executing the code

Assuming `C:/cifar-10-batches-bin` contains the dataset (`data_batch_n.bin` and `test_batch.bin`)

### Training autoencoder

`./AUTOENCODER_EXEC(_CPU) C:/cifar-10-batches-bin`

Weights will be output to `params_epoch_n.bin` in the current working directory after each epoch.

### Evaluating autoencoder

`./AUTOENCODER_EVAL(_CPU) C:/cifar-10-batches-bin`

A folder `test_outputs` will be created, containing original and reconstructed version of the first 20 images in the dataset

### Extract features for SVM (GPU only)

`./EXTRACT_TRAIN C:/cifar-10-batches-bin`

Four CSV files will be created, `train_features.csv`, `train_labels.csv`, `test_features.csv`, `test_labels.csv` containing training and testing data for SVM.
