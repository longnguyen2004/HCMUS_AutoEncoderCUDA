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
        cudaMemcpy(new_tensor.buffer.get(), this->buffer.get(), bytes, cudaMemcpyHostToDevice);
    else if (this->device == GPU && target_device == CPU)
        cudaMemcpy(new_tensor.buffer.get(), this->buffer.get(), bytes, cudaMemcpyDeviceToHost);

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
    float* dst = tensor.buffer.get();
    size_t offset_r = 3 * IMAGE_PIXELS * index;
    size_t offset_g = offset_r + IMAGE_PIXELS;
    size_t offset_b = offset_g + IMAGE_PIXELS;

    if (tensor.device == CPU) {
        std::memcpy(&dst[offset_r], r, sizeof(float) * IMAGE_PIXELS);
        std::memcpy(&dst[offset_g], g, sizeof(float) * IMAGE_PIXELS);
        std::memcpy(&dst[offset_b], b, sizeof(float) * IMAGE_PIXELS);
    }
    else {
        cudaMemcpy(&dst[offset_r], r, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice);
        cudaMemcpy(&dst[offset_g], g, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice);
        cudaMemcpy(&dst[offset_b], b, sizeof(float) * IMAGE_PIXELS, cudaMemcpyHostToDevice);     
    }
}