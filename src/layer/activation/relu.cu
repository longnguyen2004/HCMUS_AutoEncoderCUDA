#include "relu.h"
#include <helper/gpu_helper.h>
#include <algorithm>

__global__ void relu_forward_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = fmaxf(0.0f, data[idx]);
}

Tensor Relu::forward(Tensor& input) {
    if (this->cached_input != nullptr) delete this->cached_input;
    this->cached_input = new Tensor(input);

    Tensor result(input.shape, input.device);
    
    size_t input_size = input.shape.size();
    size_t bytes = input_size * sizeof(float);
    
    if(input.device == GPU) {
        CHECK(cudaMemcpy(result.data(), input.data(), bytes, cudaMemcpyDeviceToDevice));
        int gridSize = (input_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
        relu_forward_kernel<<<gridSize, BLOCK_SIZE_1D>>>(result.data(), input.shape.size());
    } else {
        for (size_t i = 0; i < input_size; ++i)
            result.data()[i] = std::max(input.data()[i], 0.0f);
    }
    return result;
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    float derivative = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    grad_input[idx] = grad_output[idx] * derivative;
}

Tensor Relu::backward(Tensor& grad_output) {
    Tensor grad_input(grad_output.shape, grad_output.device);
    size_t input_size = grad_output.shape.size();

    if (grad_output.device == GPU) {
        int gridSize = (input_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

        relu_backward_kernel<<<gridSize, BLOCK_SIZE_1D>>>(
            grad_output.data(), 
            this->cached_input->data(),
            grad_input.data(),
            input_size
        );
    }
    else {
        for (size_t i = 0; i < input_size; ++i) {
            float input_val = this->cached_input->data()[i];
            float grad_val = grad_output.data()[i];
            
            grad_input.data()[i] = (input_val > 0.0f) ? grad_val : 0.0f;
        }
    }

    return grad_input;
}
