#pragma once

#include <string.h>

#include "Config.h"

inline float* get_targets(float* targets, int digit) {
    memset(targets, 0, N_OUT * sizeof(float));
    targets[digit] = 1.0f;
    return targets;
}

inline float* get_batched_targets(float* targets, int digits[BATCH_SIZE]) {
    memset(targets, 0, N_OUT * BATCH_SIZE * sizeof(float));
    
    for (size_t b = 0; b < BATCH_SIZE; b++) {
        targets[digits[b] + b * N_OUT] = 1.0f;
    }

    return targets;
}

namespace Loss {
    float CategoricalCrossEntropy(float outputs[N_OUT], float targets[N_OUT]);
}