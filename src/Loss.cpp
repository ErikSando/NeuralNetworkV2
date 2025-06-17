#include <cmath>

#include "Config.h"
#include "Loss.h"

namespace Loss {
    float CategoricalCrossEntropy(float outputs[N_OUT], float targets[N_OUT]) {
        float sum = 0.0f;

        for (size_t i = 0; i < N_OUT; i++) {
            sum -= targets[i] * std::log(outputs[i]);
        }

        return sum;
    }
}