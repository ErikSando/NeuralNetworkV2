#include <cmath>
#include <iostream>

#include "CL/cl.h"

#include "Activation.h"
#include "Config.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Thing.h"

NeuralNetwork::NeuralNetwork() {
    kernel_mmul = new Kernel(MAT_KRNL_PATH, "Multiply");
    kernel_bmmul = new Kernel(MAT_KRNL_PATH, "MultiplyBatched");
    kernel_madd = new Kernel(MAT_KRNL_PATH, "Add");
    kernel_mscale = new Kernel(MAT_KRNL_PATH, "Scale");
    kernel_actv = new Kernel(ACTV_KRNL_PATH, "ReLU");

    Matrix::Create(nullptr, h1_nodes, BxH1);
    Matrix::Create(nullptr, h2_nodes, BxH2);
    Matrix::Create(nullptr, output_nodes, BxO);

    Matrix::Create(nullptr, h1_weights, IxH1);
    Matrix::Create(nullptr, h2_weights, H1xH2);
    Matrix::Create(nullptr, output_weights, H2xO);

    // One row is repeated for each batch, there is probably a better way to do this but I will leave it for now
    Matrix::Create(nullptr, h1_biases, BxH1);
    Matrix::Create(nullptr, h2_biases, BxH2);
    Matrix::Create(nullptr, output_biases, BxO);

    Kernel kernel_rand(MAT_KRNL_PATH, "Randomise");
    Kernel kernel_populate(MAT_KRNL_PATH, "Populate");

    float weight_max = std::sqrt(2.0f / static_cast<float>(NODE_COUNT[INPUT]));
    float weight_min = -weight_max;

    Matrix::Randomise(&kernel_rand, h1_weights, IxH1, weight_min, weight_max);
    Matrix::Randomise(&kernel_rand, h2_weights, H1xH2, weight_min, weight_max);
    Matrix::Randomise(&kernel_rand, output_weights, H2xO, weight_min, weight_max);

    Matrix::Populate(&kernel_populate, h1_biases, BxH1, 0.0f);
    Matrix::Populate(&kernel_populate, h2_biases, BxH2, 0.0f);
    Matrix::Populate(&kernel_populate, output_biases, BxO, 0.0f);
}

NeuralNetwork::~NeuralNetwork() {
    Matrix::Destroy(h1_nodes);
    Matrix::Destroy(h2_nodes);
    Matrix::Destroy(output_nodes);

    Matrix::Destroy(h1_weights);
    Matrix::Destroy(h2_weights);
    Matrix::Destroy(output_weights);

    Matrix::Destroy(h1_biases);
    Matrix::Destroy(h2_biases);
    Matrix::Destroy(output_biases);

    delete kernel_mmul;
    delete kernel_bmmul;
    delete kernel_madd;
    delete kernel_actv;
}

// I am considering changing the arguments to const float* inputs, float* outputs
cl_int NeuralNetwork::GetOutputs(
    const std::array<float, BxI>& inputs,
    std::array<float, BxO>& outputs
) {
    // this stuff with the inputs should probably be done by the function caller prior to the function call
    cl_int err;
    cl_mem input_nodes;

    err = Matrix::Create(inputs.data(), input_nodes, BxI);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to create input matrix: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    err = Matrix::Multiply(
        kernel_mmul,
        input_nodes, h1_weights, h1_nodes,
        BATCH_SIZE, NODE_COUNT[HIDDEN_1], NODE_COUNT[INPUT],
        WorkSize::Global::IH1, WorkSize::Local::IH1
    );

    err |= Matrix::Add(
        kernel_madd,
        h1_nodes, h1_biases, h1_nodes,
        BxH1
    );

    err |= Activation::ReLU(kernel_actv, h1_nodes, BxH1);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 1 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    err = Matrix::Multiply(
        kernel_mmul,
        h1_nodes, h2_weights, h2_nodes,
        BATCH_SIZE, NODE_COUNT[HIDDEN_2], NODE_COUNT[HIDDEN_1],
        WorkSize::Global::H1H2, WorkSize::Local::H1H2
    );

    err |= Matrix::Add(
        kernel_madd,
        h2_nodes, h2_biases, h2_nodes,
        BxH2
    );

    err |= Activation::ReLU(kernel_actv, h2_nodes, BxH1);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 2 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    err = Matrix::Multiply(
        kernel_mmul,
        h2_nodes, output_weights, output_nodes,
        BATCH_SIZE, NODE_COUNT[OUTPUT], NODE_COUNT[HIDDEN_2],
        WorkSize::Global::H2O, WorkSize::Local::H2O
    );

    err |= Matrix::Add(
        kernel_madd,
        output_nodes, output_biases, output_nodes,
        BxO
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute output nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    // there might be a better way to do this
    float int_output_nodes[BxO]; // intermediate output nodes, i think it makes sense to name it that
    
    err = Matrix::Transfer(
        output_nodes,
        int_output_nodes,
        BxO * sizeof(float)
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to read into intermediate output array: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    Activation::Softmax(int_output_nodes, outputs.data(), NODE_COUNT[OUTPUT], BATCH_SIZE);

    return CL_SUCCESS;
}
