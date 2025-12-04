#include "tensor.h"
#include <helper/gpu_helper.h>
#include <cstring>

size_t Shape::size() {
    return this->n * this->c * this->h * this->w;
}

Tensor::Tensor(std::shared_ptr<float[]> existing_buf, Shape s, DeviceType d) :
    buffer(existing_buf), shape(s), device(d) {
    view = tensor_view(buffer.get(), s.n, s.c, s.h, s.w);
}

Tensor::Tensor(Shape shape, DeviceType device): shape(shape), device(device) {
    if (device == CPU)
        buffer = std::shared_ptr<float[]>(new float[shape.size()]);
    else {
        float* d_data;
        CHECK(cudaMalloc(&d_data, shape.size() * sizeof(float)));
        buffer = std::shared_ptr<float[]>(d_data, [](float* ptr) { cudaFree(ptr); });
    }
    
    view = tensor_view(buffer.get(), shape.n, shape.c, shape.h, shape.w);
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

float& Tensor::operator()(size_t n, size_t c, size_t h, size_t w) {
    size_t index = 
        (((size_t)n * shape.c + c) * shape.h + h) * shape.w + w;
    return buffer[index];
}

void Tensor::from_image(Tensor& tensor, size_t index, const float* r, const float* g, const float* b, const size_t pixels) {
    size_t offset_r = 3 * pixels * index;
    size_t offset_g = offset_r + pixels;
    size_t offset_b = offset_g + pixels;
    size_t channel_size = sizeof(float) * pixels;

    if (tensor.device == CPU) {
        std::memcpy(&tensor.buffer[offset_r], r, channel_size);
        std::memcpy(&tensor.buffer[offset_g], g, channel_size);
        std::memcpy(&tensor.buffer[offset_b], b, channel_size);
    }
    else {
        CHECK(cudaMemcpy(&tensor.buffer[offset_r], r, channel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&tensor.buffer[offset_g], g, channel_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&tensor.buffer[offset_b], b, channel_size, cudaMemcpyHostToDevice));     
    }
}

float* Tensor::data() {
        return view.data_handle();
    }

const float* Tensor::data() const {
    return view.data_handle();
}