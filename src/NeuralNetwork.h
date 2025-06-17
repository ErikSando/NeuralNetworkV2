#pragma once

#include <array>

#include "CL/cl.h"

#include "Config.h"
#include "Kernel.h"

struct TestData {
    int correct;
    int incorrect;
};

class NeuralNetwork {
    public:

    NeuralNetwork();
    ~NeuralNetwork();

    // void Train(int epochs);
    // void Test(int samples);

    cl_int GetOutputs(
        const std::array<float, BxI>& inputs,
        std::array<float, BxO>& outputs
    );

    void Test(TestData& data, int samples);

    cl_mem h1_nodes;
    cl_mem h2_nodes;
    cl_mem output_nodes;

    cl_mem h1_weights;
    cl_mem h2_weights;
    cl_mem output_weights;

    cl_mem h1_biases;
    cl_mem h2_biases;
    cl_mem output_biases;

    Kernel* kernel_mscale;
    Kernel* kernel_mmul;
    Kernel* kernel_bmmul;
    Kernel* kernel_madd;
    Kernel* kernel_actv;

    int testing_row = 0;
    int training_row = 0;
};
