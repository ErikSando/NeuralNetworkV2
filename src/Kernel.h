#pragma once

#include <CL/cl.h>

class Kernel {
    public:

    Kernel(const char* kernel_path, const char* name);

    template<typename T>
    cl_int SetArgument(cl_uint index, T* value) {
        return clSetKernelArg(clkernel, index, sizeof(T), value);
    }

    bool setup_sucess = false;

    cl_program clprogram;
    cl_kernel clkernel;
};