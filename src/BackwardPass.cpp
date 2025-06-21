#include <cmath>
#include <cstring>

#include "DataParser.h"
#include "Loss.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

cl_int NeuralNetwork::BackwardPass(const cl_mem& inputs, int* digits) {
    cl_int err = GetOutputs(inputs);

    if (err != CL_SUCCESS) {
        std::cout << "Forward pass failed: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::BackwardPass)\n";
        return err;
    }

    float targets[N_OUT * BATCH_SIZE];
    get_batched_targets(targets, digits);

    cl_mem d_targets;
    err = Matrix::Transfer(targets, d_targets, N_OUT * BATCH_SIZE);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to allocate device memory for target digits: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::BackwardPass)\n";
        return err;
    }

    err = kernel_bwp->SetArgument<const cl_mem>(0, inputs);
    err |= kernel_bwp->SetArgument<const cl_mem>(1, h1_nodes);
    err |= kernel_bwp->SetArgument<const cl_mem>(2, h2_nodes);
    err |= kernel_bwp->SetArgument<const cl_mem>(3, output_nodes);
    err |= kernel_bwp->SetArgument<const cl_mem>(4, d_targets);
    err |= kernel_bwp->SetArgument<cl_mem>(5, h1_weights);
    err |= kernel_bwp->SetArgument<cl_mem>(6, h2_weights);
    err |= kernel_bwp->SetArgument<cl_mem>(7, output_weights);
    err |= kernel_bwp->SetArgument<cl_mem>(8, h1_biases);
    err |= kernel_bwp->SetArgument<cl_mem>(9, h2_biases);
    err |= kernel_bwp->SetArgument<cl_mem>(10, output_biases);
    err |= kernel_bwp->SetArgumentArray(11, N_H1);
    err |= kernel_bwp->SetArgumentArray(12, N_H2);
    err |= kernel_bwp->SetArgument<const int>(13, N_INP);
    err |= kernel_bwp->SetArgument<const int>(14, N_H1);
    err |= kernel_bwp->SetArgument<const int>(15, N_H2);
    err |= kernel_bwp->SetArgument<const int>(16, N_OUT);
    err |= kernel_bwp->SetArgument<const int>(17, BATCH_SIZE);
    err |= kernel_bwp->SetArgument<const int>(18, learning_rate);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::BackwardPass)\n";
        return err;
    }

    err = clEnqueueNDRangeKernel(
        CL::command_queue, kernel_bwp->clkernel, 2, nullptr,
        WorkSize::Global::BWP, WorkSize::Local::BWP,
        0, nullptr, nullptr
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::BackwardPass)\n";
    }

    return err;
}

cl_int NeuralNetwork::Train(const size_t epochs) {
    int n_batches = std::ceil(epochs * TRAINING_ROWS / BATCH_SIZE);

    cl_mem d_inputs;
    cl_int err = Matrix::Create(nullf, d_inputs, BxI);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to allocate device memory for inputs: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::Train)\n";
        return err;
    }

    for (int b = 0; b < n_batches; b++) {
        std::array<ImageData, BATCH_SIZE> image_data;
        DataParser::ParseBatch(testing_row, TEST_DATA_PATH, image_data, true);

        training_row = (training_row + BATCH_SIZE - 1) % TRAINING_ROWS + 1;

        // if (training_row <= BATCH_SIZE) {
        //     std::cout << "Epoch complete.\n";
        // }

        std::array<float, BxI> inputs;
        int digits[BATCH_SIZE];

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
            digits[i] = image_data[i].digit;
        }

        Matrix::Transfer(inputs.data(), d_inputs, BxI * sizeof(float));

        cl_int err = BackwardPass(d_inputs, digits);

        if (err != CL_SUCCESS) {
            std::cout << "Backward pass failed: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::Train)\n";
        }
    }

    return err;
}