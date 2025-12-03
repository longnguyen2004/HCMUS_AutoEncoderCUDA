#include "layer.h"
#include <helper/gpu_helper.h>

size_t Shape::size() {
    return this->n * this->c * this->h * this->w;
}

Tensor::Tensor(std::shared_ptr<float[]> existing_buf, Shape s, DeviceType d) :
    buffer(existing_buf), shape(s), device(d) {
    view = tensor_mdspan(buffer.get(), s.n, s.c, s.h, s.w);
}

Tensor::Tensor(Shape shape, DeviceType device): shape(shape), device(device) {
    if (device == CPU)
        buffer = std::shared_ptr<float[]>(new float[shape.size()]);
    else {
        float* d_data;
        CHECK(cudaMalloc(&d_data, shape.size() * sizeof(float)));
        buffer = std::shared_ptr<float[]>(d_data, [](float* ptr) { cudaFree(ptr); });
    }
    
    view = tensor_mdspan(buffer.get(), shape.n, shape.c, shape.h, shape.w);
}

Tensor Tensor::reshape(Shape new_shape) {
    if (this->shape.size() != new_shape.size())
        throw std::invalid_argument("Reshape: Element count mismatch.");
    return Tensor(this->buffer, new_shape, this->device);
}

Tensor Tensor::to(DeviceType target_device) {
    if (this->device == target_device) return *this; 

    Tensor new_tensor(this->shape, target_device);
    
    size_t bytes = this->shape.size() * sizeof(float);

    if (this->device == CPU && target_device == GPU)
        CHECK(cudaMemcpy(new_tensor.buffer.get(), this->buffer.get(), bytes, cudaMemcpyHostToDevice));
    else if (this->device == GPU && target_device == CPU)
        CHECK(cudaMemcpy(new_tensor.buffer.get(), this->buffer.get(), bytes, cudaMemcpyDeviceToHost));

    return new_tensor;
}

DeviceType Tensor::get_device() {
    return this->device;
}

float& Tensor::operator()(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    size_t index = 
        (((size_t)n * shape.c + c) * shape.h + h) * shape.w + w;
    return buffer[index];
}

void Tensor::from_image(Tensor& tensor, int index, const float* r, const float* g, const float* b) {
    size_t offset_r = 3 * IMAGE_PIXELS * index;
    size_t offset_g = offset_r + IMAGE_PIXELS;
    size_t offset_b = offset_g + IMAGE_PIXELS;

    if (tensor.device == CPU) {
        std::memcpy(&tensor.buffer[offset_r], r, sizeof(float) * IMAGE_PIXELS);
        std::memcpy(&tensor.buffer[offset_g], g, sizeof(float) * IMAGE_PIXELS);
        std::memcpy(&tensor.buffer[offset_b], b, sizeof(float) * IMAGE_PIXELS);
    }
    else {
        CHECK(cudaMemcpy(&tensor.buffer[offset_r], r, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&tensor.buffer[offset_g], g, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&tensor.buffer[offset_b], b, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice));     
    }
}

__global__ void relu_forward(float* data, size_t size) {
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
        CHECK(cudaMemcpy(result.buffer.get(), input.buffer.get(), bytes, cudaMemcpyDeviceToDevice));
        int gridSize = (input_size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
        relu_forward<<<gridSize, BLOCK_SIZE_1D>>>(result.buffer.get(), input.shape.size());
    } else {
        for (int i = 0; i < input_size; ++i)
            result.buffer[i] = std::max(input.buffer[i], 0.0f);
    }
    return result;
}

__global__ void relu_backward(const float* grad_output, const float* input, float* grad_input, size_t size) {
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

        relu_backward<<<gridSize, BLOCK_SIZE_1D>>>(
            grad_output.buffer.get(), 
            this->cached_input->buffer.get(),
            grad_input.buffer.get(),
            input_size
        );
    }
    else {
        for (size_t i = 0; i < input_size; ++i) {
            float input_val = this->cached_input->buffer[i];
            float grad_val = grad_output.buffer[i];
            
            grad_input.buffer[i] = (input_val > 0.0f) ? grad_val : 0.0f;
        }
    }

    return grad_input;
}