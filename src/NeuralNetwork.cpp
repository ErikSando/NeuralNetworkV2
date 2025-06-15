#include <iostream>

#include "CL/cl.h"

#include "Activation.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Thing.h"

NeuralNetwork::NeuralNetwork(Kernel& mat_mul, Kernel& batched_mat_mul, Kernel& mat_add, Kernel& activation)
    : kernel_mmul(mat_mul), kernel_bmmul(batched_mat_mul), kernel_madd(mat_add), kernel_actv(activation)
{
    Matrix::Create(nullptr, h1_nodes, BATCH_SIZE * NODE_COUNT[HIDDEN_1]);
    Matrix::Create(nullptr, h2_nodes, BATCH_SIZE * NODE_COUNT[HIDDEN_2]);
    Matrix::Create(nullptr, output_nodes, BATCH_SIZE * NODE_COUNT[OUTPUT]);

    Matrix::Create(nullptr, h1_weights, NODE_COUNT[INPUT] * NODE_COUNT[HIDDEN_1]);
    Matrix::Create(nullptr, h2_weights, NODE_COUNT[HIDDEN_1] * NODE_COUNT[HIDDEN_2]);
    Matrix::Create(nullptr, output_weights, NODE_COUNT[HIDDEN_2] * NODE_COUNT[OUTPUT]);

    // 1 row is repeated for each batch, there is probably a better way to do this but I will leave it for now
    Matrix::Create(nullptr, h1_biases, BATCH_SIZE * NODE_COUNT[HIDDEN_1]);
    Matrix::Create(nullptr, h2_biases, BATCH_SIZE * NODE_COUNT[HIDDEN_2]);
    Matrix::Create(nullptr, output_biases, BATCH_SIZE * NODE_COUNT[OUTPUT]);

    
}

NeuralNetwork::~NeuralNetwork() {
    Matrix::Destroy(h1_nodes);
    Matrix::Destroy(h2_nodes);
    Matrix::Destroy(output_nodes);

    Matrix::Destroy(h1_weights);
    Matrix::Destroy(h2_weights);
    Matrix::Destroy(output_weights);

    Matrix::Destroy(h1_biases);
    Matrix::Destroy(h2_biases);
    Matrix::Destroy(output_biases);
}

cl_int NeuralNetwork::GetOutputs(
    const std::array<float, BATCH_SIZE * NODE_COUNT[INPUT]>& inputs,
    std::array<float, BATCH_SIZE * NODE_COUNT[OUTPUT]>& outputs
) {
    cl_int err;

    cl_mem input_nodes;
    err = Matrix::Create((float*) inputs.data(), input_nodes, BATCH_SIZE * NODE_COUNT[INPUT]);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to create input matrix: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    // i think ive seen it as mat * weights but from doing the math myself I think this works
    // perhaps ive flipped the matrix layout? (or im just completely messing up)
    // H1 = ReLU(w1 I + b1)
    // H2 = ReLU(w2 H1 + b2)
    // Out = Softmax(w_out H2 + b_out)

    // err = Matrix::BatchedMultiply(kernel_bmmul, h1_weights, input_nodes, h1_nodes, NODE_COUNT[HIDDEN_1], BATCH_SIZE, NODE_COUNT[INPUT], BATCH_SIZE);
    err = Matrix::Multiply(
        kernel_mmul,
        input_nodes, h1_weights, h1_nodes,
        BATCH_SIZE, NODE_COUNT[HIDDEN_1], NODE_COUNT[INPUT],
        WorkSize::Global::IH1, WorkSize::Local::IH1
    );
    err |= Matrix::Add(
        kernel_madd,
        h1_nodes, h1_biases, h1_nodes,
        BATCH_SIZE * NODE_COUNT[HIDDEN_1]
    );
    err |= Activation::ReLU(kernel_actv, h1_nodes, BATCH_SIZE * NODE_COUNT[HIDDEN_1]);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 1 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    // err = Matrix::BatchedMultiply(kernel_bmmul, h2_weights, h1_nodes, h2_nodes, NODE_COUNT[HIDDEN_2], BATCH_SIZE, NODE_COUNT[HIDDEN_1], BATCH_SIZE);
    err = Matrix::Multiply(
        kernel_mmul,
        h1_nodes, h2_weights, h2_nodes,
        BATCH_SIZE, NODE_COUNT[HIDDEN_2], NODE_COUNT[HIDDEN_1],
        WorkSize::Global::H1H2, WorkSize::Local::H1H2
    );
    err |= Matrix::Add(
        kernel_madd,
        h2_nodes, h2_biases, h2_nodes,
        BATCH_SIZE * NODE_COUNT[HIDDEN_2]
    );
    err |= Activation::ReLU(kernel_actv, h2_nodes, BATCH_SIZE * NODE_COUNT[HIDDEN_1]);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute hidden layer 2 nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    // err = Matrix::BatchedMultiply(kernel_bmmul, output_weights, h2_nodes, output_nodes, NODE_COUNT[OUTPUT], BATCH_SIZE, NODE_COUNT[HIDDEN_2], BATCH_SIZE);
    err = Matrix::Multiply(
        kernel_mmul,
        h2_nodes, output_weights, output_nodes,
        BATCH_SIZE, NODE_COUNT[OUTPUT], NODE_COUNT[HIDDEN_2],
        WorkSize::Global::H2O, WorkSize::Local::H2O
    );
    err |= Matrix::Add(
        kernel_madd,
        output_nodes, output_biases, output_nodes,
        BATCH_SIZE * NODE_COUNT[OUTPUT]
    );

    if (err != CL_SUCCESS) {
        std::cout << "Failed to compute output nodes: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    // there might be a better way to do this
    std::array<float, BATCH_SIZE * NODE_COUNT[OUTPUT]> int_output_nodes; // intermediate output nodes, i think it makes sense to name it that
    err = Matrix::ReadInto(output_nodes, int_output_nodes.data(), BATCH_SIZE * NODE_COUNT[OUTPUT] * sizeof(float));

    if (err != CL_SUCCESS) {
        std::cout << "Failed to read into intermediate output array: " << err << " (" << FILE_NAME(__FILE__) << " > NeuralNetwork::GetOutputs)\n";
        return err;
    }

    Activation::Softmax(int_output_nodes.data(), outputs.data(), NODE_COUNT[OUTPUT], BATCH_SIZE);

    return CL_SUCCESS;
}
