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
    __global volatile atomic_float* h1_deltas,
    __global volatile atomic_float* h_deltas,
    const int n_inputs,
    const int h1_size,
    const int h2_size,
    const int n_outputs,
    const int batch_size,
    const float learning_rate
) {
    // it might be faster to find the target output here, instead of in host code
    // ill test out that once i get something working

    int digit = get_global_id(0);
    int batch = get_global_id(1);
    int node = digit + n_outputs * batch;

    if (digit >= n_outputs || batch >= batch_size) return;

    float target = targets[node];
    float output = out_nodes[node];
    float delta_out = target - output;

    for (int i = 0; i < h2_size; i++) {
        // atomic_fetch_add(&h_deltas[i], delta_out * out_weights[node + i * n_outputs]);
        out_weights[node] -= delta_out * learning_rate * h2_nodes[node + i * h2_size];
    }

    out_biases[node] -= delta_out * learning_rate;

    if (digit == 0) {
        for (int i = 0; i < h2_size; i++) {
            float delta_h2 = atomic_load(&h_deltas[i]);
            float sum = 0.0f;

            for (int j = 0; j < h1_size; j++) {
                float idk = delta_h2 * h2_weights[i + j * h2_size];
                // atomic_fetch_add(&h1_deltas[j], idk);
                h2_weights[i + batch * h2_size] -= delta_h2 * learning_rate * h1_nodes[j + batch * h1_size] / batch_size;
            }

            h2_biases[i + batch * h2_size] -= delta_h2 * learning_rate / batch_size;
        }

        for (int i = 0; i < h1_size; i++) {
            float delta_h1 = atomic_load(&h1_deltas[i]);

            for (int j = 0; j < n_inputs; j++) {
                h1_weights[i + batch * h1_size] -= delta_h1 * learning_rate * inputs[j + batch * n_inputs] / batch_size;
            }

            h1_biases[i + batch * h1_size] -= delta_h1 * learning_rate / batch_size;
        }
    }
}