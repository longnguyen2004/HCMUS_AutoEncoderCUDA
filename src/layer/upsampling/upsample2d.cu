#include "upsample2d.h"

UpSample2D::UpSample2D(int scale) : scale_factor(scale) {
}

Tensor UpSample2D::forward(Tensor& input) {
    // TODO: Implement upsampling forward pass
    return input;
}

Tensor UpSample2D::backward(Tensor& grad_output) {
    // TODO: Implement upsampling backward pass
    return grad_output;
}
