// Based on https://ecatue.gitlab.io/GPU2016/cookbook/matrix_multiplication_opencl/
// bounds checking must be done by the host code

__kernel void MatrixMultiply(
    __global const float* A, // M x K
    __global const float* B, // K x N
    __global float* C,       // M x N (output)
    const int N,             // number of columns in matrix B and C
    const int K              // number of columns in matrix A
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

__kernel void MatrixAdd(__global const float* m1, __global const float* m2, __global float* out) {
    int index = get_global_id(0);
    out[index] = m1[index] + m2[index];
}