# AutoEncoder CUDA

A C++/CUDA implementation of a convolutional autoencoder for CIFAR-10.

## 1. Hardware Requirements
- **GPU**: NVIDIA CUDA-capable GPU (tested on P100, T4, 1660 Ti).
- **Architecture**: Any modern SM architecture.
- **Memory**: ~2GB VRAM recommended.

## 2. Setup Instructions
- **CUDA**: Version 11.0+
- **Compiler**: C++20 compliant (GCC 10+, Clang 10+, or MSVC 2019+)
- **Build System**: CMake 3.18+
- **Libraries**: `nvToolsExt` (CUDA Toolkit), `mdspan` (Auto-downloaded)
- **Dataset**: [CIFAR-10 binary](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) extracted to a folder.

## 3. Compilation Commands
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```
*Note: Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU (e.g., 60, 75, 86).*

## 4. Execution Instructions
### Training
```bash
./build/AUTOENCODER_EXEC path/to/cifar-10-batches-bin
```
### Evaluation
```bash
./build/AUTOENCODER_EVAL path/to/cifar-10-batches-bin
```
### Feature Extraction
```bash
./build/EXTRACT_FEATURES path/to/cifar-10-batches-bin
```

## 5. Expected Outputs
- **Training**: Periodic loss logs in console and `params_epoch_n.bin` weight files.
- **Evaluation**: A `test_outputs` folder containing original vs. reconstructed PNG images.
- **Feature Extraction**: Four CSV files (`train_features.csv`, etc.) for SVM training.
