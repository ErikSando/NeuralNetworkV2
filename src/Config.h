#pragma once

#include <cstddef>

constexpr bool GPU = true;

constexpr size_t TRAINING_ROWS = 60000;
constexpr size_t TESTING_ROWS = 10000;

constexpr const char* MAT_KRNL_PATH = "src/MatrixKernel.cl";
constexpr const char* ACTV_KRNL_PATH = "src/ActivationKernel.cl";

constexpr size_t BATCH_SIZE = 32;
constexpr size_t NODE_COUNT[4] = { 28 * 28, 128, 64, 10 };

enum Layer {
    INPUT, HIDDEN_1, HIDDEN_2, OUTPUT
};

constexpr size_t IxH1 = NODE_COUNT[INPUT] * NODE_COUNT[HIDDEN_1];
constexpr size_t H1xH2 = NODE_COUNT[HIDDEN_1] * NODE_COUNT[HIDDEN_2];
constexpr size_t H2xO = NODE_COUNT[HIDDEN_2] * NODE_COUNT[OUTPUT];

constexpr size_t BxI = BATCH_SIZE * NODE_COUNT[INPUT];
constexpr size_t BxH1 = BATCH_SIZE * NODE_COUNT[HIDDEN_1];
constexpr size_t BxH2 = BATCH_SIZE * NODE_COUNT[HIDDEN_2];
constexpr size_t BxO = BATCH_SIZE * NODE_COUNT[OUTPUT];

constexpr size_t round_up(size_t lws, size_t dim) {
    return ((dim + lws - 1) / lws) * lws;
}

constexpr int LWS = 16; // default local work size

namespace WorkSize {
    namespace Local {
        constexpr size_t IH1[2] = { LWS, LWS };
        constexpr size_t H1H2[2] = { LWS, LWS };
        constexpr size_t H2O[2] = { LWS, LWS };
    }
    
    namespace Global {
        constexpr size_t IH1[2] = {
            round_up(Local::IH1[0], NODE_COUNT[HIDDEN_1]),
            round_up(Local::IH1[1], BATCH_SIZE)
        };

        constexpr size_t H1H2[2] = {
            round_up(Local::H1H2[0], NODE_COUNT[HIDDEN_2]),
            round_up(Local::H1H2[1], BATCH_SIZE)
        };

        constexpr size_t H2O[2] = { // water reference?
            round_up(Local::H2O[0], NODE_COUNT[OUTPUT]),
            round_up(Local::H2O[1], BATCH_SIZE)
        };
    }
}