// Based on https://ecatue.gitlab.io/GPU2016/cookbook/_multiplication_opencl/

__kernel void Multiply(
    __global const float* A, // M x K
    __global const float* B, // K x N
    __global float* C,       // M x N (output)
    const int M,             // number of rows in  A
    const int N,             // number of columns in  B and C
    const int K              // number of columns in  A
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// I don't think I need this one
__kernel void MultiplyBatched(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const int batch_size
) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int batch = get_global_id(2);

    if (row >= M || col >= N || batch >= batch_size) return;

    int A_offset = batch * M * K;
    int B_offset = batch * K * N;
    int C_offset = batch * M * N;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[row * K + k + A_offset] * B[k * N + col + B_offset];
    }

    C[row * N + col + C_offset] = sum;
}

__kernel void Add(__global const float* A, __global const float* B, __global float* C) {
    int index = get_global_id(0);
    C[index] = A[index] + B[index];
}

__kernel void Scale(__global float* mat, const float k) {
    int index = get_global_id(0);
    mat[index] *= k;
}

__kernel void Populate(__global float* mat, float value) {
    int index = get_global_id(0);
    mat[index] = value;
}

uint xorshift(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

float randf(uint seed) {
    uint rand = xorshift(seed);
    return convert_float(rand) * (1.0 / 4294967296.0f); // compiler will precompute this division at compile time, i think
}

__kernel void Randomise(__global float* mat, float min, float max, uint seed) {
    int index = get_global_id(0);
    float rand = randf(seed + (uint) index * 1234567) * (max - min) + min;
    mat[index] = rand;
}