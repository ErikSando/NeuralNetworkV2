#include <cmath>
#include <cstring>

#include "DataParser.h"
#include "Loss.h"
#include "NeuralNetwork.h"

void NeuralNetwork::Train(const size_t epochs) {
    int n_batches = std::ceil(epochs * TRAINING_ROWS / BATCH_SIZE);

    for (int b = 0; b < n_batches; b++) {
        std::array<ImageData, BATCH_SIZE> image_data;
        DataParser::ParseBatch(testing_row, TEST_DATA_PATH, image_data, true);

        training_row = (training_row + BATCH_SIZE - 1) % TRAINING_ROWS + 1;

        // if (training_row <= BATCH_SIZE) {
        //     std::cout << "Epoch complete.\n";
        // }

        std::array<float, BxI> inputs;
        std::array<float, BxO> outputs;

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
        }

        GetOutputs(inputs, outputs);

        // i think what you do is
        // go through and decide how much each weight/bias should change
        // average these amounts over each sample
        // apply changes (will need to use GPU to modify cl_mem's)

        // ill probably need to use the GPU

        // change in weights and biases that will be applied later
        float h1_deltas[IxH1] = {0};
        float h2_deltas[H1xH2] = {0};
        float out_deltas[H2xO] = {0};

        for (int i = 0; i < BATCH_SIZE; i++) {
            float output_errors[N_OUT];
            float targets[N_OUT];

            get_targets(targets, image_data[i].digit);

            for (size_t d = 0; d < N_OUT; d++) {
                output_errors[d] = targets[d] - outputs[d + i * N_OUT];
            }

            
        }

        // std::array<float, N_OUT> output_deltas;
        // std::array<float, N_OUT> targets = get_targets();

        // for (size_t d = 0; d < N_OUT; d++) {
        //     output_deltas[d] = ;
        // }
    }
}