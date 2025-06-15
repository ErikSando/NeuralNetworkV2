#pragma once

#include "CL/cl.h"

#include "Kernel.h"

namespace Activation {
    cl_int ReLU(Kernel& kernel, cl_mem& mat, const size_t len);

    void Softmax(const float* inp, float* out, const int C, const int batches);
}