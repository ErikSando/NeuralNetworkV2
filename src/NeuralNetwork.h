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

    // used when the output is to be sent to the CPU
    cl_int GetOutputs(
        const std::array<float, BxI>& inputs,
        std::array<float, BxO>& outputs
    );

    cl_int GetOutputs(const std::array<float, BxI>& inputs);

    void Train(const size_t epochs);
    void Test(TestData& test_data, int batches);

    float learning_rate = 0.005f;

    int testing_row = 0;
    int training_row = 0;

    // private:

    cl_int ForwardPass(const std::array<float, BxI>& inputs);

    //void CheckOutputs(cl_mem& test_data);

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
    Kernel* kernel_oactv; // output activation
};
