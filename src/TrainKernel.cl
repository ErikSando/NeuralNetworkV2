// #include <opencl_atomic>

// this function uses a 1 dimensional global work size, but maybe it could be made
// more efficient with more dimensions
// I will try that out once I get it working

__kernel void BackwardPass(
    __global float* const inputs,
    __global float* const nodes_h1,
    __global float* const nodes_h2,
    __global float* const nodes_out,
    __global float* const targets,
    __global float* weights_h1,
    __global float* weights_h2,
    __global float* weights_out,
    __global float* biases_h1,
    __global float* biases_h2,
    __global float* biases_out,
    // __global float* w_gradients_h1,
    // __global float* w_gradients_h2,
    // __global float* w_gradients_out,
    // __global float* b_gradients_h1,
    // __global float* b_gradients_h2,
    // __global float* b_gradients_out,
    __global volatile atomic_float* deltas_h1,
    __global volatile atomic_float* deltas_h2,
    const int n_inputs,
    const int size_h1,
    const int size_h2,
    const int n_outputs,
    const int batch_size,
    const float learning_rate
) {
    // one thread for each output node i guess
    // it might be faster to find the target output here, instead of in host code
    // ill test out that once i get something working

    // because the bias matrix needs to have identical rows, take the average
    // gradient of each column and add that to each element in the colum

    int digit = get_global_id(0);
    int batch = get_global_id(1);
    int node = digit + n_outputs * batch;

    if (digit >= n_outputs || batch >= batch_size) return;

    float target = targets[node];
    float output = nodes_out[node];
    float delta_out = target - output;

    for (int i = 0; i < size_h2; i++) {
        //atomic_fetch_add(&deltas_h2[i], delta_out * weights_out[node + i * n_outputs]);
        // w_gradients_out[node] -= delta_out * nodes_h2[node + i * size_h2];
        weights_out[node] -= delta_out * learning_rate * nodes_h2[node + i * size_h2];
    }

    // b_gradients_out[node] -= delta_out;
    biases_out[node] -= delta_out * learning_rate;

    if (digit == 0) {
        for (int i = 0; i < size_h2; i++) {
            float delta_h2 = atomic_load(&deltas_h2[i]);
            float sum = 0.0f;

            for (int j = 0; j < size_h1; j++) {
                float idk = delta_h2 * weights_h2[i + j * size_h2];
                //atomic_fetch_add(&deltas_h1[j], idk);
                // atomic_fetch_sub(&w_gradients_h2[i + j * size_h2], delta_h2 * nodes_h1[j + batch * size_h1]);
                weights_h2[i + batch * size_h2] -= delta_h2 * learning_rate * nodes_h1[j + batch * size_h1] / batch_size;
            }

            // atomic_fetch_add(&deltas_h1[i], sum);

            // b_gradients_h2[i] -= delta_h2;
            biases_h2[i + batch * size_h2] -= delta_h2 * learning_rate / batch_size;
        }

        for (int i = 0; i < size_h1; i++) {
            float delta_h1 = atomic_load(&deltas_h1[i]);

            for (int j = 0; j < n_inputs; j++) {
                // atomic_fetch_sub(&w_gradients_h1[i + j * size_h1], delta_h1 * inputs[j + batch * n_inputs]);
                weights_h1[i + batch * size_h1] -= delta_h1 * learning_rate * inputs[j + batch * n_inputs] / batch_size;
            }

            // b_gradients_h1[i] -= delta_h1;
            biases_h1[i + batch * size_h1] -= delta_h1 * learning_rate / batch_size;
        }
    }
}