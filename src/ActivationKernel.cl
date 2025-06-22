__kernel void ReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0;
}

__kernel void LeakyReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0.01 * M[index];
}

__kernel void Softmax(
    __global float* M,
    __local float* exponents,
    const int C,
    const int batch_size
) {
    int index = get_global_id(0);
    int batch = get_global_id(1);

    if (index >= C || batch >= batch_size) return;

    exponents[index + batch * C] = exp(M[index + batch * C]);

    barrier(CLK_GLOBAL_MEM_FENCE);

    float sum = 0.0f;

    if (index == 0) {
        for (int i = 0; i < 10; i++) {
            sum += exponents[i];
        }

        exponents[batch_size * (C + 1)] = sum; // 32 elements at the end are the 32 sums
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    M[index + batch * C] = exponents[index + batch * C] / exponents[batch_size * C + batch];
}