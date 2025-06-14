#include <iostream>

#include <CL/cl.h>

#include "Thing.h"

namespace CL {
    cl_platform_id platform_ids[100];
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;

    bool setup_success = false;

    cl_int Init() {
        cl_int err;
        cl_uint num_platforms;

        err = clGetPlatformIDs(0, nullptr, &num_platforms);

        if (err != CL_SUCCESS) {
            std::cout << "clGetPlatformIDs error: " << err << "\n";
            return 1;
        }

        std::cout << num_platforms << " platforms/s.\n";

        err = clGetPlatformIDs(num_platforms, platform_ids, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "clGetPlatformIDs error: " << err << "\n";
            return 1;
        }

        err = clGetDeviceIDs(platform_ids[0], GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "clGetDeviceIDs error: " << err << "\n";
            return 1;
        }

        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);

        if (!context) {
            std::cout << "clCreateContext error: " << err << "\n";
            return 1;
        }

        command_queue = clCreateCommandQueue(context, device_id, 0, &err);

        if (!command_queue) {
            std::cout << "clCreateCommandQueue error: " << err << "\n";
            return 1;
        }

        return CL_SUCCESS;
    }

    void Destroy() {
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }
}