#include "../output.h"
#include <constants.h>
#include <helper/gpu_helper.h>
#include <stdexcept>
#include <utility>
#include <vector>

__global__ void mse_grad_kernel(const float* pred, const float* target, float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = (2.0f / static_cast<float>(n)) * (pred[idx] - target[idx]);
    }
}

__global__ void mse_loss_accum_kernel(const float* pred, const float* target, float* loss_accum, int n) {
    // Block-level reduction of squared error; accumulate to global loss_accum.
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (idx < n) {
        float d = pred[idx] - target[idx];
        diff = d * d;
    }
    sdata[threadIdx.x] = diff;
    __syncthreads();

    // Reduce within the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss_accum, sdata[0]);
    }
}

OutputGPU::OutputGPU(std::shared_ptr<Layer> prev) {
    m_prev = std::move(prev);
    auto [x, y, z] = dimension();
    const size_t n = static_cast<size_t>(x) * y * z;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&grad_input), n * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_loss), sizeof(float)));
}

OutputGPU::~OutputGPU() {
    CHECK(cudaFree(d_target));
    CHECK(cudaFree(d_loss));
}

std::tuple<int, int, int> OutputGPU::dimension() const {
    return m_prev->dimension();
}

const float* OutputGPU::output() const {
    return m_prev->output();
}

float OutputGPU::loss() const {
    return m_loss;
}

void OutputGPU::setReferenceImage(const std::vector<float>& image) {
    auto [x, y, z] = dimension();
    const size_t expected = static_cast<size_t>(x) * y * z;
    if (image.size() != expected) {
        throw std::runtime_error("OutputGPU: reference image size does not match network output");
    }

    h_target = image;

    if (!d_target) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_target), expected * sizeof(float)));
    }
    CHECK(cudaMemcpy(d_target, h_target.data(), expected * sizeof(float), cudaMemcpyHostToDevice));
}

void OutputGPU::forward() {
    if (h_target.empty()) {
        throw std::runtime_error("OutputGPU: reference image not set");
    }

    m_prev->forward();
    auto [x, y, z] = dimension();
    const size_t n = static_cast<size_t>(x) * y * z;

    // Accumulate loss on device to avoid large host transfers.
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((static_cast<int>(n) + blockSize.x - 1) / blockSize.x);
    mse_loss_accum_kernel<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(m_prev->output(), d_target, d_loss, static_cast<int>(n));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    float h_loss = 0.0f;
    CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    m_loss = h_loss / static_cast<float>(n);
}

void OutputGPU::backward(float learning_rate, const float* /*grad_output*/) {
    if (h_target.empty() || !d_target) {
        throw std::runtime_error("OutputGPU: reference image not set");
    }

    auto [x, y, z] = dimension();
    const int n = x * y * z;

    dim3 blockSize(BLOCK_SIZE_1D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    mse_grad_kernel<<<gridSize, blockSize>>>(m_prev->output(), d_target, grad_input, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    m_prev->backward(learning_rate, grad_input);
}
