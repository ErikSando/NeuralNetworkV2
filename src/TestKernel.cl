// This isn't being used currently

#define N_OUT 10

__kernel void CheckOutputs(
    __global const float* outputs,
    __global const int* digits,
    __global int* test_data
) {
    int batch_idx = get_global_id(0);
    int index = batch_idx * N_OUT;

    float max = 0.0f;
    int digit = -1;

    for (int i = 0; i < N_OUT; i++) {
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