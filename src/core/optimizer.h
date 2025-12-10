#pragma once

// CPU version: Update parameters with gradient descent
void updateWeightsCPU(float* params, const float* gradients, float learning_rate, int n);

// GPU version: Update parameters with gradient descent
void updateWeightsGPU(float* params, const float* gradients, float learning_rate, int n);

// CPU version: Clip gradients by value
void clipGradientsCPU(float* gradients, float clip_value, int n);

// GPU version: Clip gradients by value
void clipGradientsGPU(float* gradients, float clip_value, int n);

// GPU version: Clip gradients by global norm
void clipGradientsByNormGPU(float* gradients, float max_norm, int n);
