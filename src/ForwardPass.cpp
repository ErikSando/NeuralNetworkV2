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
        kernels[MMUL],
        inputs, h1_weights, h1_nodes,
        BATCH_SIZE, N_H1, N_INP,
        WorkSize::Global::IH1, WorkSize::Local::IH1
    );

    err |= Matrix::Add(
        kernels[MADD],
        h1_nodes, h1_biases, h1_nodes,
        BxH1
    );

    err |= Activation::ReLU(kernels[ACTV], h1_nodes, BxH1);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to compute hidden layer 1 nodes", err);
        return err;
    }

    err = Matrix::Multiply(
        kernels[MMUL],
        h1_nodes, h2_weights, h2_nodes,
        BATCH_SIZE, N_H2, N_H1,
        WorkSize::Global::H1H2, WorkSize::Local::H1H2
    );

    err |= Matrix::Add(
        kernels[MADD],
        h2_nodes, h2_biases, h2_nodes,
        BxH2
    );

    err |= Activation::ReLU(kernels[ACTV], h2_nodes, BxH1);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to compute hidden layer 2 nodes", err);
        return err;
    }

    err = Matrix::Multiply(
        kernels[MMUL],
        h2_nodes, output_weights, output_nodes,
        BATCH_SIZE, N_OUT, N_H2,
        WorkSize::Global::H2O, WorkSize::Local::H2O
    );

    err |= Matrix::Add(
        kernels[MADD],
        output_nodes, output_biases, output_nodes,
        BxO
    );

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to compute output nodes", err);
    }

    return err;
}

// Runs forward pass but then applies soft max
cl_int NeuralNetwork::GetOutputs(const cl_mem& inputs) {
    cl_int err = ForwardPass(inputs);

    if (err != CL_SUCCESS) {
        return err;
    }

    err = Activation::Softmax(kernels[OACTV], output_nodes, N_OUT, BATCH_SIZE);

    if (err != CL_SUCCESS) {
        ERROR_CL("Softmax failed", err);
    }

    return err;
}

cl_int NeuralNetwork::GetOutputs(const std::array<float, BxI>& inputs) {
    cl_mem d_inputs;
    cl_int err = Matrix::Create(inputs.data(), d_inputs, BxI);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to allocate device memory for inputs", err);
        Matrix::Destroy(d_inputs);
        return err;
    }

    err = ForwardPass(d_inputs);
    Matrix::Destroy(d_inputs);

    if (err != CL_SUCCESS) {
        ERROR_CL("Forward pass failed", err);
    }

    // err = Activation::Softmax(kernels[OACTV], output_nodes, N_OUT, BATCH_SIZE);

    // if (err != CL_SUCCESS) {
    //     ERROR_CL("Softmax failed", err);
    // }

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
        BxO
    );

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to read into output array", err);
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
    cl_int err = Matrix::Create(inputs.data(), d_inputs, BxI);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to allocate device memory for inputs", err);
        Matrix::Destroy(d_inputs);
    }

    err = ForwardPass(d_inputs);
    Matrix::Destroy(d_inputs);

    if (err != CL_SUCCESS) {
        ERROR_CL("Forward pass failed", err);
    }

    err = Matrix::Transfer(
        output_nodes,
        outputs.data(),
        BxO
    );

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to read into output array", err);
        return err;
    }

    Activation::Softmax(outputs.data(), N_OUT, BATCH_SIZE);

    return err;
}