#include "optimizer.h"
#include <algorithm>
#include <cmath>

void updateWeightsCPU(float* params, const float* gradients, float learning_rate, int n) {
    for (int i = 0; i < n; ++i) {
        params[i] -= learning_rate * gradients[i];
    }
}

void clipGradientsCPU(float* gradients, float clip_value, int n) {
    for (int i = 0; i < n; ++i) {
        gradients[i] = std::max(-clip_value, std::min(clip_value, gradients[i]));
    }
}
