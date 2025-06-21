#include <cmath>

#include <iostream>

#include "CL/cl.h"

#include "Activation.h"
#include "Config.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Thing.h"

cl_int NeuralNetwork::ForwardPass(const cl_mem& inputs) {
    // this stuff with the inputs should probably be done by the function caller prior to the function call
    cl_int err;

    err = Matrix::Multiply(
        kernel_mmul,
        inputs, h1_weights, h1_nodes,
        BATCH_SIZE, N_H1, N_INP,
        WorkSize::Global::IH1, WorkSize::Local::IH1
    );

    err |= Matrix::Add(
        kernel_madd,
        h1_nodes, h1_biases, h1_nodes,
        BxH1
    );

    err |= Activation::ReLU(kernel_actv, h1_nodes, BxH1);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 1 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::ForwardPass)\n";
        return err;
    }

    err = Matrix::Multiply(
        kernel_mmul,
        h1_nodes, h2_weights, h2_nodes,
        BATCH_SIZE, N_H2, N_H1,
        WorkSize::Global::H1H2, WorkSize::Local::H1H2
    );

    err |= Matrix::Add(
        kernel_madd,
        h2_nodes, h2_biases, h2_nodes,
        BxH2
    );

    err |= Activation::ReLU(kernel_actv, h2_nodes, BxH1);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 2 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::ForwardPass)\n";
        return err;
    }

    err = Matrix::Multiply(
        kernel_mmul,
        h2_nodes, output_weights, output_nodes,
        BATCH_SIZE, N_OUT, N_H2,
        WorkSize::Global::H2O, WorkSize::Local::H2O
    );

    err |= Matrix::Add(
        kernel_madd,
        output_nodes, output_biases, output_nodes,
        BxO
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute output nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::ForwardPass)\n";
    }

    return err;
}

// Runs forward pass but then applies soft max
cl_int NeuralNetwork::GetOutputs(const cl_mem& inputs) {
    cl_int err = ForwardPass(inputs);

    if (err != CL_SUCCESS) {
        return err;
    }

    // this softmax is very slow!
    err = Activation::Softmax(kernel_oactv, output_nodes, N_OUT, BATCH_SIZE);

    return err;
}

cl_int NeuralNetwork::GetOutputs(const std::array<float, BxI>& inputs) {
    cl_mem d_inputs;
    cl_int err = Matrix::Transfer(inputs.data(), d_inputs, BxI * sizeof(float));

    if (err != CL_SUCCESS) {
        std::cout << "Failed to allocate device memory for inputs: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        Matrix::Destroy(d_inputs);
    }

    err = ForwardPass(d_inputs);
    Matrix::Destroy(d_inputs);

    if (err != CL_SUCCESS) {
        std::cout << "Forward pass failed: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
    }

    return err;
}

cl_int NeuralNetwork::GetOutputs(
    const cl_mem& inputs,
    std::array<float, BxO>& outputs
) {
    cl_int err = ForwardPass(inputs);

    if (err != CL_SUCCESS) {
        return err;
    }

    err = Matrix::Transfer(
        output_nodes,
        outputs.data(),
        BxO * sizeof(float)
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to read into output array: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    Activation::Softmax(outputs.data(), N_OUT, BATCH_SIZE);

    return err;
}

cl_int NeuralNetwork::GetOutputs(
    const std::array<float, BxI>& inputs,
    std::array<float, BxO>& outputs
) {
    cl_mem d_inputs;
    cl_int err = Matrix::Transfer(inputs.data(), d_inputs, BxI * sizeof(float));

    if (err != CL_SUCCESS) {
        std::cout << "Failed to allocate device memory for inputs: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        Matrix::Destroy(d_inputs);
    }

    err = ForwardPass(d_inputs);
    Matrix::Destroy(d_inputs);

    if (err != CL_SUCCESS) {
        std::cout << "Forward pass failed: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
    }

    err = Matrix::Transfer(
        output_nodes,
        outputs.data(),
        BxO * sizeof(float)
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to read into output array: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    Activation::Softmax(outputs.data(), N_OUT, BATCH_SIZE);

    return err;
}