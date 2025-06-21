#pragma once

#include "CL/cl.h"

class Kernel {
    public:

    Kernel(const char* kernel_path, const char* name);
    ~Kernel();

    template<typename T>
    cl_int SetArgument(cl_uint index, const T& value) {
        return clSetKernelArg(clkernel, index, sizeof(T), &value);
    }

    cl_int SetArgumentArray(cl_uint index, const int size) {
        return clSetKernelArg(clkernel, index, size, nullptr);
    }

    cl_program clprogram;
    cl_kernel clkernel;
};