// #include <opencl_atomic>

// try to change the delta_h1 and delta_h2 writes to have no conflicts
// cos trying to use atomic addition with floats seemingly doesn't want to work

__kernel void BackwardPass(
    __global float* const inputs,
    __global float* const h1_nodes,
    __global float* const h2_nodes,
    __global float* const out_nodes,
    __global float* const targets,
    __global float* h1_weights,
    __global float* h2_weights,
    __global float* out_weights,
    __global float* h1_biases,
    __global float* h2_biases,
    __global float* out_biases,
    __local volatile atomic_float* h1_deltas,
    __local volatile atomic_float* h_deltas,
    const int n_inputs,
    const int h1_size,
    const int h2_size,
    const int n_outputs,
    const int batch_size,
    const float learning_rate
) {
    
}