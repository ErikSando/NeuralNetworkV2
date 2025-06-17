#include <array>
#include <iostream>
#include <string.h>

#include "DataParser.h"
#include "Loss.h"
#include "NeuralNetwork.h"

void NeuralNetwork::Test(TestData& test_data, int batches) {
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

        GetOutputs(inputs, outputs);

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
}
