#pragma once

// CPU version: Update parameters with gradient descent
void updateWeightsCPU(float* params, const float* gradients, float learning_rate, int n);

// GPU version: Update parameters with gradient descent
void updateWeightsGPU(float* params, const float* gradients, float learning_rate, int n);
