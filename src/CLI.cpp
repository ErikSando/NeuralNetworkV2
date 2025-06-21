#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>

#include "Config.h"
#include "DataParser.h"
#include "FileInput.h"
#include "Thing.h"
#include "NeuralNetwork.h"

#include "Matrix.h"

float randf() {
    return (float) rand() / (float) RAND_MAX;
}

int num_characters(int n) {
    if (n == 0) return 1;

    return (n > 0 ? 1 : 2) + std::floor(std::log10(std::abs(n)));
}

void PrintData(ImageData& data, int spacing = 4) {
    std::cout << "Digit: " << data.digit << std::endl;

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            int pixel = (float) (data.pixels[r * 28 + c]) * 255.0f;
            int n_digits = num_characters(pixel);

            for (int i = 0; i < spacing - n_digits; i++) {
                std::cout << " ";
            }

            std::cout << pixel;
        }

        std::cout << std::endl;
    }
}

int CommandLoop() {
    NeuralNetwork network;

    std::string command;

    while (true) {
        std::cout << "> ";
        std::getline(std::cin, command);

        std::istringstream iss(command);
        std::vector<std::string> args;
        std::string arg;

        while (iss >> arg) {
            args.emplace_back(arg);
        }

        if (args.size() < 1) continue;

        std::string& cmd = args[0];

        if (cmd == "exit" || cmd == "quit") {
            break;
        }
        else if (cmd == "help") {
            std::cout << "help\n - Shows this menu.\n";
            std::cout << "train [no. epochs]\n - Train the network with the specified number of epochs. One epoch uses the entire training data set.\n";
            std::cout << "test [no. batches]\n - Test the networks accuracy for the specified number of batches. Type 'all' to use the entire testing data set.\n";
            // std::cout << "id [file path]\n - Identify a digit in a 28x28 drawing.\n";
        }
        else if (cmd == "getout") {
            std::array<float, BxI> inputs;
            std::array<float, BxO> outputs;

            for (size_t i = 0; i < BATCH_SIZE * N_INP; i++) {
                inputs[i] = randf();// * 255;
            }

            network.GetOutputs(inputs, outputs);

            // only the first set of output nodes are displayed
            for (size_t i = 0; i < N_OUT; i++) {
                std::cout << i << ": " << outputs[i] << "\n";
            }
        }
        else if (cmd == "idrand") {
            std::array<ImageData, BATCH_SIZE> image_data;

            int line = random() % TESTING_ROWS;

            DataParser::ParseBatch(line, "res/mnistdata/mnist_test.csv", image_data);

            PrintData(image_data[0]);

            std::array<float, BxI> inputs;
            float outputs[BxO];

            for (size_t i = 0; i < BATCH_SIZE; i++) {
                memcpy(image_data[i].pixels.begin(), inputs.begin() + i * N_INP, N_INP * sizeof(float));
            }

            network.GetOutputs(inputs);

            Matrix::Transfer(network.output_nodes, outputs, N_OUT * sizeof(float));

            for (size_t i = 0; i < N_OUT; i++) {
                std::cout << i << ": " << outputs[i] << "\n";
            }
        }
        else if (cmd == "test") {
            if (args.size() < 2) {
                std::cout << "Incorrect usage\n";
                std::cout << "Usage: train [no. batches]\n";
                continue;
            }

            std::string& batches_arg = args[1];
            int batches = 0;

            if (batches_arg == "all") batches = TESTING_ROWS / BATCH_SIZE;
            else batches = std::stoi(batches_arg);

            if (!batches) continue;

            std::cout << "Testing...\n";
            
            TestData test_data;

            auto start = std::chrono::high_resolution_clock::now();

            network.Test(test_data, batches);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            int correct = test_data.correct;
            int incorrect = test_data.incorrect;

            float accuracy = 100 * static_cast<float>(correct) / static_cast<float>(correct + incorrect);

            std::cout << "Testing complete in " << duration << " ms.\n";
            std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << (correct + incorrect) << ")\n";
        }
    }

    return 0;
}