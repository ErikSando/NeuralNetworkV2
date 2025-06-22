#include <array>
#include <cstring>
#include <iostream>

#include "CL/cl.h"

#include "DataParser.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

cl_int NeuralNetwork::Test(TestData& test_data, int batches) {
    test_data.correct = 0;
    test_data.incorrect = 0;

    for (int b = 0; b < batches; b++) {
        std::array<ImageData, BATCH_SIZE> image_data;
        DataParser::ParseBatch(testing_row, TEST_DATA_PATH, image_data, true);

        testing_row = (testing_row + BATCH_SIZE - 1) % TESTING_ROWS + 1;

        std::array<float, BxI> inputs;
        std::array<float, BxO> outputs;

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
        }

        cl_int err = GetOutputs(inputs, outputs);

        if (err != CL_SUCCESS) {
            ERROR_CL("Forward pass failed", err);
            continue;
        }

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            float largest_output = 0;
            int prediction = -1;

            for (size_t d = 0; d < N_OUT; d++) {
                float output = outputs[i * N_OUT + d];

                if (output >= largest_output) {
                    largest_output = output;
                    prediction = d;
                }
            }

            if (prediction == image_data[i].digit) test_data.correct++;
            else test_data.incorrect++;
        }
    }

    return CL_SUCCESS;
}

// Doing everything on the GPU like below is about as fast but the accuracy is different
// so I think I messed something up, I think the code above is the correct one
// #include <array>
// #include <cstring>
// #include <iostream>

// #include "CL/cl.h"

// #include "DataParser.h"
// #include "Loss.h"
// #include "Matrix.h"
// #include "NeuralNetwork.h"

// cl_int CheckOutputs(Kernel& kernel, const cl_mem& outputs, const cl_mem& digits, cl_mem& test_data) {
//     cl_int err;

//     err  = kernel.SetArgument<const cl_mem>(0, outputs);
//     err |= kernel.SetArgument<const cl_mem>(1, digits);
//     err |= kernel.SetArgument<cl_mem>(2, test_data);
//     err |= kernel.SetArgument<const int>(3, N_OUT);

//     if (err != CL_SUCCESS) {
//         ERROR_CL("Failed to set kernel arguments", err);
//         return err;
//     }

//     err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &BATCH_SIZE, nullptr, 0, nullptr, nullptr);

//     if (err != CL_SUCCESS) {
//         ERROR_CL("Failed to execute kernel", err);
//     }

//     return err;
// }

// cl_int NeuralNetwork::Test(TestData& test_data, int batches) {
//     test_data.correct = 0;
//     test_data.incorrect = 0;

//     Kernel check_kernel("src/TestKernel.cl", "CheckOutputs");

//     int test_data_buf[2] = { 0, 0 };

//     cl_int err;

//     cl_mem test_data_cl; // { correct, incorrect }
//     err = Matrix::Create(test_data_buf, test_data_cl, 2);

//     if (err != CL_SUCCESS) {
//         ERROR_CL("Failed to create device buffer for test data", err);
//         Matrix::Destroy(test_data_cl);
//         return err;
//     }

//     cl_mem d_digits;
//     err = Matrix::Create((int*) nullptr, d_digits, BATCH_SIZE);

//     if (err != CL_SUCCESS) {
//         ERROR_CL("Failed to create device buffer for digits", err);
//         Matrix::Destroy(test_data_cl);
//         Matrix::Destroy(d_digits);
//         return err;
//     }

//     for (int b = 0; b < batches; b++) {
//         std::array<ImageData, BATCH_SIZE> image_data;
//         DataParser::ParseBatch(testing_row, TEST_DATA_PATH, image_data, true);

//         testing_row = (testing_row + BATCH_SIZE - 1) % TESTING_ROWS + 1;

//         int digits[BATCH_SIZE];

//         std::array<float, BxI> inputs;

//         for (size_t i = 0; i < BATCH_SIZE; i++) {
//             memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
//             digits[i] = image_data[i].digit;
//         }

//         err = Matrix::Transfer(digits, d_digits, BATCH_SIZE);

//         if (err != CL_SUCCESS) {
//             ERROR_CL("Failed to copy digits into device memory", err);
//             continue;
//         }

//         err = GetOutputs(inputs);

//         if (err != CL_SUCCESS) {
//             ERROR_CL("Forward pass failed", err);
//             continue;
//         }

//         err = CheckOutputs(check_kernel, output_nodes, d_digits, test_data_cl);

//         if (err != CL_SUCCESS) {
//             ERROR_CL("Failed to check outputs", err);
//         }
//     }

//     Matrix::Transfer(test_data_cl, test_data_buf, 2);

//     Matrix::Destroy(test_data_cl);
//     Matrix::Destroy(d_digits);

//     test_data.correct = test_data_buf[0];
//     test_data.incorrect = test_data_buf[1];

//     // test_data.correct = 0;
//     // test_data.incorrect = 0;

//     return CL_SUCCESS;
// }