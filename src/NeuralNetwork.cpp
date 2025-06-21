#include <cmath>
#include <ctime>
#include <random>

#include "Config.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    srand(time(nullptr));

    kernel_mmul = new Kernel(MAT_KRNL_PATH, "Multiply");
    kernel_bmmul = new Kernel(MAT_KRNL_PATH, "MultiplyBatched");
    kernel_madd = new Kernel(MAT_KRNL_PATH, "Add");
    kernel_mscale = new Kernel(MAT_KRNL_PATH, "Scale");
    kernel_actv = new Kernel(ACTV_KRNL_PATH, "ReLU");
    kernel_oactv = new Kernel(ACTV_KRNL_PATH, "Softmax");
    kernel_bwp = new Kernel(BWP_KRNL_PATH, "BackwardPass");

    Matrix::Create(nullf, h1_nodes, BxH1);
    Matrix::Create(nullf, h2_nodes, BxH2);
    Matrix::Create(nullf, output_nodes, BxO);

    Matrix::Create(nullf, h1_weights, IxH1);
    Matrix::Create(nullf, h2_weights, H1xH2);
    Matrix::Create(nullf, output_weights, H2xO);

    // One row is repeated for each batch, there is probably a better way to do this but I will leave it for now
    Matrix::Create(nullf, h1_biases, BxH1);
    Matrix::Create(nullf, h2_biases, BxH2);
    Matrix::Create(nullf, output_biases, BxO);

    Kernel kernel_rand(MAT_KRNL_PATH, "Randomise");
    Kernel kernel_populate(MAT_KRNL_PATH, "Populate");

    float weight_max = std::sqrt(2.0f / static_cast<float>(N_INP));
    float weight_min = -weight_max;

    Matrix::Randomise(&kernel_rand, h1_weights, IxH1, weight_min, weight_max);
    Matrix::Randomise(&kernel_rand, h2_weights, H1xH2, weight_min, weight_max);
    Matrix::Randomise(&kernel_rand, output_weights, H2xO, weight_min, weight_max);

    Matrix::Populate(&kernel_populate, h1_biases, BxH1, 0.0f);
    Matrix::Populate(&kernel_populate, h2_biases, BxH2, 0.0f);
    Matrix::Populate(&kernel_populate, output_biases, BxO, 0.0f);
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

    delete kernel_mmul;
    delete kernel_bmmul;
    delete kernel_madd;
    delete kernel_actv;
}