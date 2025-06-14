#include <fstream>
#include <iostream>
#include <sstream>

#include <CL/cl.h>

#include "Kernel.h"
#include "Thing.h"

const char* read_file(const char* file_path) {
    std::ifstream file;
    file.open(file_path, std::ios::binary | std::ios::ate);

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
    const char* kernel_src = read_file(kernel_path);

    if (!kernel_src) {
        delete[] kernel_src;
        std::cout << "Failed to create kernel: failed to read kernel source\n";
        return;
    }

    cl_int err;

    clprogram = clCreateProgramWithSource(CL::context, 1, (const char**) &kernel_src, nullptr, &err);

    delete[] kernel_src;

    if (!clprogram) {
        std::cout << "Failed to create compute program: " << err << "\n";
        return;
    }

    err = clBuildProgram(clprogram, 0, nullptr, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(clprogram, CL::device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cout << "Failed to build program executible.\n";
        std::cout << buffer << "\n";
        return;
    }

    clkernel = clCreateKernel(clprogram, name, &err);

    if (!clkernel || err != CL_SUCCESS) {
        std::cout << "Failed to create compute kernel.\n";
        return;
    }

    setup_sucess = true;
}