#include <iostream>

#include <CL/cl.h>

#include "Kernel.h"
#include "Matrix.h"
#include "Thing.h"

void clean_up(cl_mem& dmA, cl_mem& dmB, cl_mem& dmC) {
    Matrix::Destroy(dmA); // are checked to be non-null in Matrix::Destroy
    Matrix::Destroy(dmB);
    Matrix::Destroy(dmC);
    CL::Destroy();
}

int main() {
    CL::Init();

    if (!CL::setup_success) {
        std::cout << "Failed to intialise OpenCL.\n";
        return 1;
    }

    Kernel kernel("src/MatrixKernel.cl", "MatrixMultiply");

    if (!kernel.setup_sucess) {
        std::cout << "Failed to create kernel.\n";
        return 1;
    }

    const float A[9] = {
        3.0f, 1.0f, 2.0f,
        2.0f, 3.0f, 1.0f,
        1.0f, 2.0f, 3.0f
    };

    const float B[9] = {
        1.0f, 3.0f, 2.0f,
        2.0f, 1.0f, 3.0f,
        3.0f, 2.0f, 1.0f
    };

    float C[9];

    cl_mem dmA;
    cl_mem dmB;
    cl_mem dmC;

    cl_int err;

    err = Matrix::Create(A, &dmA, 9);
    err |= Matrix::Create(B, &dmB, 9);
    err |= Matrix::Create(C, &dmC, 9);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to create matrices.\n";
        clean_up(dmA, dmB, dmC);
        return 1;
    }

    err = Matrix::Multiply(kernel, &dmA, &dmB, &dmC, C, 3, 3);

    clean_up(dmA, dmB, dmC);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to multiply matrices.\n";
        return 1;
    }

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            std::cout << " " << C[r * 3 + c];
        }

        std::cout << "\n";
    }

    return 0;
}