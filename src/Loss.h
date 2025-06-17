#pragma once

#include <string.h>

#include "Config.h"

inline float* get_targets(float* targets, int digit) {
    memset(targets, 0, N_OUT * sizeof(float));
    targets[digit] = 1.0f;
    return targets;
}

namespace Loss {
    float CategoricalCrossEntropy(float outputs[N_OUT], float targets[N_OUT]);
}