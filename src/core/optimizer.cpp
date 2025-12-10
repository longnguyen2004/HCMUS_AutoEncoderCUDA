#include "optimizer.h"

void updateWeightsCPU(float* params, const float* gradients, float learning_rate, int n) {
    for (int i = 0; i < n; ++i) {
        params[i] -= learning_rate * gradients[i];
    }
}
