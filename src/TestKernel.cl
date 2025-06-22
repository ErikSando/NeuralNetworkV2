__kernel void CheckOutputs(
    __global const float* outputs,
    __global const int* digits,
    __global int* test_data,
    const int C
) {
    int batch = get_global_id(0);
    int index = batch * C;

    float max = -2000000000.0f;
    int digit = -1;

    for (int i = 0; i < C; i++) {
        float output = outputs[index + i];

        if (output > max) {
            max = output;
            digit = i;
        }
    }

    if (digit == digits[batch]) atomic_inc(&test_data[0]);
    else atomic_inc(&test_data[1]);
}