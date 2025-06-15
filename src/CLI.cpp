#include <array>
#include <iostream>
#include <sstream>
#include <vector>

#include <ctime>
#include <random>

#include "Thing.h"
#include "NeuralNetwork.h"

int CommandLoop() {
    srand(time(nullptr));

    Kernel kernel_mmul("src/MatrixKernel.cl", "MatrixMultiply");
    Kernel kernel_bmmul("src/MatrixKernel.cl", "BatchedMatrixMultiply");
    Kernel kernel_madd("src/MatrixKernel.cl", "MatrixAdd");
    Kernel kernel_actv("src/ActivationKernel.cl", "ReLU");

    if (!kernel_mmul.clkernel || !kernel_bmmul.clkernel || !kernel_madd.clkernel || !kernel_actv.clkernel) {
        std::cout << "Failed to create kernel (" << FILE_NAME(__FILE__) << ")\n";
        return 1;
    }

    NeuralNetwork network(kernel_mmul, kernel_bmmul, kernel_madd, kernel_actv);

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
            std::array<float, BATCH_SIZE * NODE_COUNT[INPUT]> inputs;
            std::array<float, BATCH_SIZE * NODE_COUNT[OUTPUT]> outputs;

            for (size_t i = 0; i < BATCH_SIZE * NODE_COUNT[INPUT]; i++) {
                inputs[i] = rand();
            }

            network.GetOutputs(inputs, outputs);

            for (size_t i = 0; i < NODE_COUNT[OUTPUT]; i++) {
                std::cout << i << ": " << outputs[i] << "\n";
            }
        }
    }

    return 0;
}