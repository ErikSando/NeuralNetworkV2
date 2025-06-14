#include <chrono>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

#include <CL/cl.h>

#include "Kernel.h"
#include "Matrix.h"
#include "Thing.h"

#include "CPUMM.h"

void clean_up(cl_mem& dmA, cl_mem& dmB, cl_mem& dmC, float* A, float* B, float* C, float* C2 = nullptr) {
    Matrix::Destroy(dmA); // are checked to be non-null in Matrix::Destroy
    Matrix::Destroy(dmB);
    Matrix::Destroy(dmC);
    CL::Destroy();
    delete[] A;
    delete[] B;
    delete[] C;
    if (C2) delete[] C2;
}

int main() {
    srand(time(nullptr));

    cl_int err = CL::Init();

    if (err != CL_SUCCESS) {
        std::cout << "Failed to intialise OpenCL.\n";
        return 1;
    }

    Kernel kernel_mm("src/MatrixKernel.cl", "MatrixMultiply");
    Kernel kernel_bmm("src/MatrixKernel.cl", "BatchedMatrixMultiply");

    if (!kernel_mm.clkernel) {
        std::cout << "Failed to create kernel.\n";
        return 1;
    }

    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    float k = 1e-10f;

    for (int i = 0; i < M * K; i++) {
        A[i] = (float) (rand()) * k - k / 2;
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = (float) (rand()) * k - k / 2;
    }

    cl_mem dmA;
    cl_mem dmB;
    cl_mem dmC;

    err = Matrix::Create(A, &dmA, M * K);
    err |= Matrix::Create(B, &dmB, K * N);
    err |= Matrix::Create(C, &dmC, M * N);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to create matrices.\n";
        clean_up(dmA, dmB, dmC, A, B, C);
        return 1;
    }

    int n = 400;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        err = Matrix::Multiply(kernel_mm, &dmA, &dmB, &dmC, C, M, N, K);
    }

    if (err != CL_SUCCESS) {
        std::cout << "Failed to multiply matrices.\n";
        clean_up(dmA, dmB, dmC, A, B, C);
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << n << " GPU matrix multiplication/s in " << duration << " ms.\n";

    Matrix::Destroy(dmA);
    Matrix::Destroy(dmB);
    Matrix::Destroy(dmC);
    delete[] A;
    delete[] B;
    delete[] C;

    A = new float[n * M * K];
    B = new float[n * K * N];
    C = new float[n * M * N];

    for (int i = 0; i < n * M * K; i++) {
        A[i] = (float) (rand()) * k - k / 2;
    }

    for (int i = 0; i < n * K * N; i++) {
        B[i] = (float) (rand()) * k - k / 2;
    }

    err = Matrix::Create(A, &dmA, n * M * K);
    err |= Matrix::Create(B, &dmB, n * K * N);
    err |= Matrix::Create(C, &dmC, n * M * N);

    if (err != CL_SUCCESS) {
        std::cout << "Failed to create matrices.\n";
        clean_up(dmA, dmB, dmC, A, B, C);
        return 1;
    }

    start = std::chrono::high_resolution_clock::now();
    err = Matrix::BatchedMultiply(kernel_bmm, &dmA, &dmB, &dmC, C, M, N, K, (size_t) n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (err != CL_SUCCESS) {
        std::cout << "Failed to batch multiply matrices.\n";
        clean_up(dmA, dmB, dmC, A, B, C);
        return 1;
    }

    std::cout << n << " batch size GPU matrix multiplication in " << duration << " ms.\n";

    start = std::chrono::high_resolution_clock::now();

    float* C2 = new float[M * N];

    for (int i = 0; i < n; i++) {
        CPUMM(A, B, C2, M, N, K);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << n << " CPU matrix multiplication/s in " << duration << " ms.\n";

    // for (int r = 0; r < M; r++) {
    //     for (int c = 0; c < N; c++) {
    //         std::cout << " " << C[r * N + c];
    //     }

    //     std::cout << "\n";
    // }

    // std::cout << "\n";

    // for (int r = 0; r < M; r++) {
    //     for (int c = 0; c < N; c++) {
    //         std::cout << " " << C2[r * N + c];
    //     }

    //     std::cout << "\n";
    // }

    clean_up(dmA, dmB, dmC, A, B, C, C2);

    return 0;
}