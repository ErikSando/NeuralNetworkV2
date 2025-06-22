#pragma once

#include <cstddef>

constexpr bool GPU = true;

constexpr size_t TRAINING_ROWS = 60000;
constexpr size_t TESTING_ROWS = 10000;

constexpr const char* TRAIN_DATA_PATH = "res/mnistdata/mnist_train.csv";
constexpr const char* TEST_DATA_PATH = "res/mnistdata/mnist_test.csv";

constexpr const char* MAT_KRNL_PATH = "src/MatrixKernel.cl";
constexpr const char* ACTV_KRNL_PATH = "src/ActivationKernel.cl";
constexpr const char* BWP_KRNL_PATH = "src/TrainKernel.cl";

constexpr size_t BATCH_SIZE = 32; // Number of samples per batch
constexpr size_t NODE_COUNT[4] = { 28 * 28, 128, 64, 10 };

enum Layer {
    INPUT, HIDDEN_1, HIDDEN_2, OUTPUT
};

constexpr size_t N_INP = NODE_COUNT[INPUT]; // Number of input nodes
constexpr size_t N_H1 = NODE_COUNT[HIDDEN_1]; // Number of hidden layer 1 nodes
constexpr size_t N_H2 = NODE_COUNT[HIDDEN_2]; // Number of hidden layer 2 nodes
constexpr size_t N_OUT = NODE_COUNT[OUTPUT]; // Number of output nodes

constexpr size_t IxH1 = N_INP * N_H1; // Number of input nodes * number of hidden layer 1 nodes
constexpr size_t H1xH2 = N_H1 * N_H2; // Number of hidden layer 1 nodes * number of hidden layer 2 nodes
constexpr size_t H2xO = N_H2 * N_OUT; // Number of hidden layer 2 nodes * number of output nodes

constexpr size_t BxI = BATCH_SIZE * N_INP; // Batch size * number of input nodes
constexpr size_t BxH1 = BATCH_SIZE * N_H1; // Batch size * number of hidden layer 1 nodes
constexpr size_t BxH2 = BATCH_SIZE * N_H2; // Batch size * number of hidden layer 2 nodes
constexpr size_t BxO = BATCH_SIZE * N_OUT; // Batch size * number of output nodes

/**
 * Calculate a global work size based on the local work size and minimum required size
 * lws - local work size
 * dim - minimum required global work size
 */
constexpr size_t round_up(size_t lws, size_t dim) {
    return ((dim + lws - 1) / lws) * lws;
}

constexpr int DLWS = 16; // Default local work size

namespace WorkSize { // Namespace containing local and global work sizes for different functions/algorithms
    namespace Local { // Namespace containing local work sizes for different functions/algorithms
        // Local work size for multiplying the input matrix by the hidden layer 1 weights
        constexpr size_t IH1[2] = { DLWS, DLWS };

        // Local work size for multiplying the hidden layer 1 matrix by the hidden layer 2 weights
        constexpr size_t H1H2[2] = { DLWS, DLWS };
        
        //Local work size for multiplying the hidden layer 2 matrix by the output weights
        constexpr size_t H2O[2] = { DLWS, DLWS };
        
        //Local work size for softmax
        constexpr size_t SM[2] = { DLWS, DLWS };
        
        // Local work size for a backward pass
        constexpr size_t BWP[2] = { DLWS, DLWS };
    }

    namespace Global { // Namespace containing global work sizes for different functions/algorithms
        // Global work size for multiplying the input matrix by the hidden layer 1 weights
        constexpr size_t IH1[2] = {
            round_up(Local::IH1[0], N_H1),
            round_up(Local::IH1[1], BATCH_SIZE)
        };

        // Global work size for multiplying the hidden layer 1 matrix by the hidden layer 2 weights
        constexpr size_t H1H2[2] = {
            round_up(Local::H1H2[0], N_H2),
            round_up(Local::H1H2[1], BATCH_SIZE)
        };

        // Global work size for multiplying the hidden layer 2 matrix by the output weights
        constexpr size_t H2O[2] = {
            round_up(Local::H2O[0], N_OUT),
            round_up(Local::H2O[1], BATCH_SIZE)
        };

        // Global work size for softmax
        constexpr size_t SM[2] = {
            round_up(Local::SM[0], N_OUT),
            round_up(Local::SM[0], BATCH_SIZE)
        };

        // Global work size for a backward pass
        constexpr size_t BWP[2] = {
            round_up(Local::BWP[0], N_OUT),
            round_up(Local::BWP[0], BATCH_SIZE)
        };
    }
}