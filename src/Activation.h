#pragma once

#include "CL/cl.h"

#include "Kernel.h"

namespace Activation {
    cl_int ReLU(Kernel* kernel, cl_mem& mem, const size_t size);

    cl_int Softmax(Kernel* kernel, cl_mem& mem, const size_t C, const int batches); // GPU
    
    void Softmax(float* mem, const int C, const int batches); // CPU
}