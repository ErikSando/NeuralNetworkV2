#include <iostream>

#include "CL/cl.h"

#include "Config.h"
#include "Thing.h"

namespace CL {
    cl_platform_info platform_info;
    cl_platform_id platform_ids[100];
    cl_device_id device_id;
    cl_ulong global_mem_size;
    cl_ulong max_alloc_size;
    cl_context context;
    cl_command_queue command_queue;

    bool setup_success = false;

    cl_int Init() {
        cl_int err;
        cl_uint num_platforms;

        err = clGetPlatformIDs(0, nullptr, &num_platforms);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to get platform IDs", err);
            return 1;
        }

        std::cout << num_platforms << " platforms/s\n";

        err = clGetPlatformIDs(num_platforms, platform_ids, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to get platform IDs", err);
            return 1;
        }

        char version[128];
        clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
        std::cout << "OpenCL version: " << version << "\n";

        err = clGetDeviceIDs(platform_ids[0], GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to get device IDs", err);
            return 1;
        }

        clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc_size), &max_alloc_size, nullptr);

        std::cout << "Global memory size: " << global_mem_size << "\n";
        std::cout << "Max mem alloc size: " << max_alloc_size << "\n";

        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);

        if (!context) {
            ERROR_CL("Failed to create context", err);
            clReleaseContext(context);
            return 1;
        }

        command_queue = clCreateCommandQueue(context, device_id, 0, &err);

        if (!command_queue) {
            ERROR_CL("Failed to create command queue", err);
            Destroy();
            return 1;
        }

        return CL_SUCCESS;
    }

    void Destroy() {
        if (command_queue) clReleaseCommandQueue(command_queue);
        if (context) clReleaseContext(context);
    }
}