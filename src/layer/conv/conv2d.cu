#include "conv2d.h"

Conv2D::Conv2D(int in_channels, int out_channels) 
    : c_in(in_channels), c_out(out_channels),
      weight(Shape{0, 0, 0, 0}, CPU),  // Placeholder initialization
      bias(Shape{0, 0, 0, 0}, CPU) {   // Placeholder initialization
    // TODO: Implement proper weight and bias initialization
}

Tensor Conv2D::forward(Tensor& input) {
    // TODO: Implement convolution forward pass
    return input;
}

Tensor Conv2D::backward(Tensor& grad_output) {
    // TODO: Implement convolution backward pass
    return grad_output;
}

void Conv2D::step(float lr) {
    // TODO: Implement weight update step
}
