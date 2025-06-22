#include <cmath>
#include <cstring>

#include "DataParser.h"
#include "Loss.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

// test if re-using cl_mem objects can improve speed

cl_int NeuralNetwork::BackwardPass(const cl_mem& inputs, int* digits) {
    cl_int err = GetOutputs(inputs);

    if (err != CL_SUCCESS) {
        ERROR_CL("Forward pass failed", err);
        return err;
    }

    float targets[N_OUT * BATCH_SIZE];
    get_batched_targets(targets, digits);

    cl_mem d_targets;
    err = Matrix::Create(targets, d_targets, N_OUT * BATCH_SIZE);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to allocate device memory for target digits", err);
        return err;
    }

    Kernel& kernel = kernels[BWP];

    err  = kernel.SetArgument(0, inputs);
    err |= kernel.SetArgument(1, h1_nodes);
    err |= kernel.SetArgument(2, h2_nodes);
    err |= kernel.SetArgument(3, output_nodes);
    err |= kernel.SetArgument(4, d_targets);
    err |= kernel.SetArgument(5, h1_weights);
    err |= kernel.SetArgument(6, h2_weights);
    err |= kernel.SetArgument(7, output_weights);
    err |= kernel.SetArgument(8, h1_biases);
    err |= kernel.SetArgument(9, h2_biases);
    err |= kernel.SetArgument(10, output_biases);
    err |= kernel.SetLocalArrayArgument<float>(11, N_H1);
    err |= kernel.SetLocalArrayArgument<float>(12, N_H2);
    err |= kernel.SetArgument(13, N_INP);
    err |= kernel.SetArgument(14, N_H1);
    err |= kernel.SetArgument(15, N_H2);
    err |= kernel.SetArgument(16, N_OUT);
    err |= kernel.SetArgument(17, BATCH_SIZE);
    err |= kernel.SetArgument(18, learning_rate);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to set kernel arguments", err);
        Matrix::Destroy(d_targets);
        return err;
    }

    err = clEnqueueNDRangeKernel(
        CL::command_queue, kernel.clkernel, 2, nullptr,
        WorkSize::Global::BWP, WorkSize::Local::BWP,
        0, nullptr, nullptr
    );

    Matrix::Destroy(d_targets);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to execute kernel", err);
    }

    return err;
}

cl_int NeuralNetwork::Train(const size_t epochs) {
    int n_batches = std::ceil(epochs * TRAINING_ROWS / BATCH_SIZE);

    cl_mem d_inputs;
    cl_int err = Matrix::Create(nullf, d_inputs, BxI);

    if (err != CL_SUCCESS) {
        ERROR_CL("Failed to allocate device memory for inputs", err);
        Matrix::Destroy(d_inputs);
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

        Matrix::Transfer(inputs.data(), d_inputs, BxI);

        cl_int err = BackwardPass(d_inputs, digits);

        if (err != CL_SUCCESS) {
            ERROR_CL("Backward pass failed", err);
        }

        std::cout << b << "\n";
    }

    Matrix::Destroy(d_inputs);

    return CL_SUCCESS;
}