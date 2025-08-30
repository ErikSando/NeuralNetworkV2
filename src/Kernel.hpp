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

    template<typename T>
    cl_int SetLocalArrayArgument(cl_uint index, const int size) {
        return clSetKernelArg(clkernel, index, size * sizeof(T), nullptr);
    }

    cl_program clprogram = nullptr;
    cl_kernel clkernel = nullptr;

    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    Kernel(Kernel&&) noexcept;
    Kernel& operator=(Kernel&&) noexcept;
};