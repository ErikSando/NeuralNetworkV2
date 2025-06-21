#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "CL/cl.h"

#include "Kernel.h"
#include "Thing.h"

const char* read_file(const char* file_path) {
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

// std::string read_file(const char* file_path) {
//     std::ifstream file(file_path);

//     if (!file.is_open()) {
//         std::cerr << "Cannot open file: " << file_path << std::endl;
//         return "";
//     }

//     std::stringstream contents;
//     contents << file.rdbuf();
//     return contents.str();
// }

Kernel::Kernel(const char* kernel_path, const char* name) {
    const char* kernel_src = read_file(kernel_path);

    // std::string kernel_src = "";
    
    // for (int i = 0; i < n_paths; i++) {
    //     kernel_src += read_file(kernel_paths[i]);
    // }

    // if (kernel_src == "") {
    if (!kernel_src) {
        std::cout << "Failed to read kernel sources (Kernel::Kernel)\n";
        return;
    }

    // const char* ksrc = kernel_src.c_str();

    cl_int err;

    // clprogram = clCreateProgramWithSource(CL::context, 1, &ksrc, nullptr, &err);
    clprogram = clCreateProgramWithSource(CL::context, 1, &kernel_src, nullptr, &err);

    if (!clprogram) {
        std::cout << "Failed to create compute program: " << err << " (" << FILE_NAME(__FILE__) << " > Kernel::Kernel)\n";
        return;
    }

    err = clBuildProgram(clprogram, 0, nullptr, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // size_t length;
        // char buffer[2048];
        // clGetProgramBuildInfo(clprogram, CL::device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cout << "Failed to build program executible: " << err << " (" << FILE_NAME(__FILE__) << " > Kernel::Kernel)\n";
        // std::cout << buffer << "\n";
        return;
    }

    clkernel = clCreateKernel(clprogram, name, &err);

    if (!clkernel || err != CL_SUCCESS) {
        std::cout << "Failed to create compute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > Kernel::Kernel)\n";
        return;
    }
}

Kernel::~Kernel() {
    clReleaseKernel(clkernel);
    clReleaseProgram(clprogram);
}