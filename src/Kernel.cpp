#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "CL/cl.h"

#include "Kernel.h"
#include "Thing.h"

char* read_file(const char* file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << file_path << std::endl;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* buffer = new char[size + 1];

    if (!file.read(buffer, size)) {
        delete[] buffer;
        std::cout << "Failed to read file: " << file_path << std::endl;
        return nullptr;
    }

    buffer[size] = '\0';

    return buffer;
}

Kernel::Kernel(const char* kernel_path, const char* name) {
    char* kernel_src = read_file(kernel_path);

    if (!kernel_src) {
        ERROR("Failed to read kernel sources");
        return;
    }

    cl_int err;

    clprogram = clCreateProgramWithSource(CL::context, 1, (const char**) &kernel_src, nullptr, &err);

    delete[] kernel_src;

    if (!clprogram) {
        ERROR_CL("Failed to create compute program", err);
        return;
    }

    err = clBuildProgram(clprogram, 0, nullptr, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // size_t length;
        // char buffer[2048];
        // clGetProgramBuildInfo(clprogram, CL::device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        ERROR_CL("Failed to build program executible", err);
        // std::cout << buffer << "\n";
        return;
    }

    clkernel = clCreateKernel(clprogram, name, &err);

    if (!clkernel || err != CL_SUCCESS) {
        ERROR_CL("Failed to create compute kernel", err);
        return;
    }
}

Kernel::~Kernel() {
    if (clkernel) clReleaseKernel(clkernel);
    if (clprogram) clReleaseProgram(clprogram);
}

Kernel::Kernel(Kernel&& other) noexcept {
    clkernel = other.clkernel;
    clprogram = other.clprogram;
    other.clkernel = nullptr;
    other.clprogram = nullptr;
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        if (clkernel) clReleaseKernel(clkernel);
        if (clprogram) clReleaseProgram(clprogram);

        clkernel = other.clkernel;
        clprogram = other.clprogram;
        other.clkernel = nullptr;
        other.clprogram = nullptr;
    }

    return *this;
}