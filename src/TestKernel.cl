// This isn't being used currently

__kernel void CheckOutputs(
    __global const float* outputs,
    __global const int* digits,
    __global int* test_data,
    const int n_outputs
) {
    int batch_idx = get_global_id(0);
    int index = batch_idx * n_outputs;

    float max = 0.0f;
    int digit = -1;

    for (int i = 0; i < n_outputs; i++) {
        float output = outputs[index + i];

        if (output > max) {
            max = output;
            digit = i;
        }
    }

    // if (digit == digits[batch_idx]) test_data[0]++;
    // else test_data[1]++;

    test_data[1]++;
}