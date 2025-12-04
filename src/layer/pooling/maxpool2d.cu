#include "maxpool2d.h"

MaxPool2D::MaxPool2D(int pool_size, int stride) 
    : pool_size(pool_size), stride(stride) {
}

Tensor MaxPool2D::forward(Tensor& input) {
    // TODO: Implement max pooling forward pass
    return input;
}

Tensor MaxPool2D::backward(Tensor& grad_output) {
    // TODO: Implement max pooling backward pass
    return grad_output;
}
