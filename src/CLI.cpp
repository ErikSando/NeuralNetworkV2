#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>

#include <ctime>
#include <random>

#include "Config.h"
#include "DataParser.h"
#include "FileInput.h"
#include "Thing.h"
#include "NeuralNetwork.h"

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
    srand(time(nullptr));

    // Kernel kernel_mmul("src/MatrixKernel.cl", "MatrixMultiply");
    // Kernel kernel_bmmul("src/MatrixKernel.cl", "BatchedMatrixMultiply");
    // Kernel kernel_madd("src/MatrixKernel.cl", "MatrixAdd");
    // Kernel kernel_actv("src/ActivationKernel.cl", "ReLU");

    // if (!kernel_mmul.clkernel || !kernel_bmmul.clkernel || !kernel_madd.clkernel || !kernel_actv.clkernel) {
    //     std::cout << "Failed to create kernel (" << FILE_NAME(__FILE__) << ")\n";
    //     return 1;
    // }

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

        std::string& cmd = args.at(0);

        if (cmd == "exit" || cmd == "quit") {
            break;
        }
        else if (cmd == "help") {
            std::cout << "help:\n- Shows this menu.\n";
        }
        else if (cmd == "getout") {
            std::array<float, BxI> inputs;
            std::array<float, BxO> outputs;

            for (size_t i = 0; i < BATCH_SIZE * NODE_COUNT[INPUT]; i++) {
                inputs[i] = randf();// * 255;
            }

            network.GetOutputs(inputs, outputs);

            // only the first set of output nodes are displayed
            for (size_t i = 0; i < NODE_COUNT[OUTPUT]; i++) {
                std::cout << i << ": " << outputs[i] << "\n";
            }
        }
        else if (cmd == "idrand") {
            std::array<ImageData, BATCH_SIZE> image_data;

            DataParser::ParseBatch(1, "res/mnistdata/mnist_train.csv", image_data);

            PrintData(image_data[0]);

            std::array<float, BxI> inputs;
            std::array<float, BxO> outputs;

            for (size_t i = 0; i < BATCH_SIZE; i++) {
                memcpy(image_data[i].pixels.begin(), inputs.begin() + i * NODE_COUNT[INPUT], NODE_COUNT[INPUT] * sizeof(float));
            }

            network.GetOutputs(inputs, outputs);

            for (size_t i = 0; i < NODE_COUNT[OUTPUT]; i++) {
                std::cout << i << ": " << outputs[i] << "\n";
            }
        }
    }

    return 0;
}