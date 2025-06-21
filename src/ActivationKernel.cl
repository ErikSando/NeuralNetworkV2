__kernel void ReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0;
}

__kernel void LeakyReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0.01 * M[index];
}

// I don't know how to make the length of the sum array change
// depending on what N_OUT is in the host code, so I am using 10
#define N_OUT 10

__kernel void Softmax(__global float* M, const int start) {
    int index = get_global_id(0);

    __local float exponents[N_OUT + 1];

    exponents[index] = exp(M[index + start]);

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;

    if (index == 0) {
        for (int i = 0; i < 10; i++) {
            sum += exponents[i];
        }

        exponents[N_OUT] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    M[index + start] = exponents[index] / exponents[N_OUT];
}