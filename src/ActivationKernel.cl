__kernel void ReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0;
}

__kernel void LeakyReLU(__global float* M) {
    int index = get_global_id(0);
    M[index] = M[index] >= 0 ? M[index] : 0.01 * M[index];
}