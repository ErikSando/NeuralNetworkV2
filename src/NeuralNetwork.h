#pragma once

#include <array>

#include "CL/cl.h"
#include "Kernel.h"

enum Layer {
    INPUT, HIDDEN_1, HIDDEN_2, OUTPUT
};

constexpr size_t BATCH_SIZE = 32;
constexpr size_t NODE_COUNT[4] = { 28 * 28, 128, 64, 10 };

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

class NeuralNetwork {
    public:

    NeuralNetwork(Kernel& mat_mul, Kernel& batched_mat_mul, Kernel& mat_add, Kernel& activation);
    ~NeuralNetwork();

    // void Train(int epochs);
    // void Test(int samples);

    cl_int GetOutputs(
        const std::array<float, BATCH_SIZE * NODE_COUNT[INPUT]>& inputs,
        std::array<float, BATCH_SIZE * NODE_COUNT[OUTPUT]>& outputs
    );

    cl_mem h1_nodes;
    cl_mem h2_nodes;
    cl_mem output_nodes;

    cl_mem h1_weights;
    cl_mem h2_weights;
    cl_mem output_weights;

    cl_mem h1_biases;
    cl_mem h2_biases;
    cl_mem output_biases;

    Kernel& kernel_mmul;
    Kernel& kernel_bmmul;
    Kernel& kernel_madd;
    Kernel& kernel_actv;
};