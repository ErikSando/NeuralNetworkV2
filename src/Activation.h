#pragma once

#include "CL/cl.h"

#include "Kernel.h"

namespace Activation {
    cl_int ReLU(Kernel& kernel, cl_mem& mem, const size_t size);

    // GPU computed softmax
    cl_int Softmax(
        Kernel& kernel, cl_mem& mem,
        const size_t C, const size_t batch_size
    );

    // CPU computed softmax
    void Softmax(float* mem, const int C, const int batches);
}