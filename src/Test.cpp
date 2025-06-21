#include <array>
#include <cstring>
#include <iostream>

#include "CL/cl.h"

#include "DataParser.h"
#include "Loss.h"
// #include "Matrix.h"
#include "NeuralNetwork.h"

// cl_int CheckOutputs(Kernel& kernel, const cl_mem& outputs, const cl_mem& digits, cl_mem& test_data) {
//     cl_int err;

//     err = kernel.SetArgument<const cl_mem>(0, outputs);
//     err |= kernel.SetArgument<const cl_mem>(1, digits);
//     err |= kernel.SetArgument<cl_mem>(2, test_data);

//     if (err != CL_SUCCESS) {
//         std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > CheckOutputs)\n";
//         return err;
//     }

//     err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &BATCH_SIZE, nullptr, 0, nullptr, nullptr);

//     if (err != CL_SUCCESS) {
//         std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > CheckOutputs)\n";
//     }

//     return err;
// }

void NeuralNetwork::Test(TestData& test_data, int batches) {
    test_data.correct = 0;
    test_data.incorrect = 0;

    // Kernel check_kernel("src/TestKernel.cl", "CheckOutputs");

    // int test_data_buf[2] = { 0, 0 };

    // cl_mem test_data_cl; // { correct, incorrect }
    // Matrix::Create(test_data_buf, test_data_cl, 2);

    // cl_mem digits;
    // Matrix::Create((int*) nullptr, digits, BATCH_SIZE);

    for (int b = 0; b < batches; b++) {
        std::array<ImageData, BATCH_SIZE> image_data;
        DataParser::ParseBatch(testing_row, TEST_DATA_PATH, image_data, true);

        testing_row = (testing_row + BATCH_SIZE - 1) % TESTING_ROWS + 1;

        // int _digits[BATCH_SIZE];

        std::array<float, BxI> inputs;
        std::array<float, BxO> outputs;

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
            // _digits[i] = image_data[i].digit;
        }

        // Matrix::Transfer(_digits, digits, BATCH_SIZE * sizeof(int));

        // GetOutputs(inputs) != CL_SUCCESS || // cheeky trick
        // CheckOutputs(check_kernel, output_nodes, digits, test_data_cl);

        if (GetOutputs(inputs, outputs) != CL_SUCCESS) continue;

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

    // Matrix::Transfer(test_data_cl, test_data_buf, 2 * sizeof(float));

    // test_data.correct = test_data_buf[0];
    // test_data.incorrect = test_data_buf[1];

    // test_data.correct = 0;
    // test_data.incorrect = 0;
}