#include <iostream>

#include <CL/cl.h>

#include "Kernel.h"
#include "Matrix.h"
#include "Thing.h"

namespace Matrix {
    constexpr size_t local_work_size[2] = { 16, 16 };
    constexpr size_t global_work_size[2] = { N, M };

    cl_int Create(const float* mat, cl_mem* dev_mem, const int size) {
        cl_int err;

        *dev_mem = clCreateBuffer(CL::context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(float), (void*) mat, &err);

        if (err != CL_SUCCESS || !dev_mem) {
            std::cout << "Failed to allocate device memory: " << err << "\n";
            return err;
        }

        return CL_SUCCESS;
    }

    void Destroy(cl_mem& dev_mem) {
        if (dev_mem) clReleaseMemObject(dev_mem);
    }

    cl_int Multiply(
        Kernel& kernel,
        const cl_mem* dmA, const cl_mem* dmB, cl_mem* dmC,
        float* C,
        const int M, const int N, const int K
    ) {
        cl_int err;

        err = kernel.SetArgument<const cl_mem>(0, *dmA);
        err |= kernel.SetArgument<const cl_mem>(1, *dmB);
        err |= kernel.SetArgument<cl_mem>(2, *dmC);
        err |= kernel.SetArgument<const int>(3, M);
        err |= kernel.SetArgument<const int>(4, N);
        err |= kernel.SetArgument<const int>(5, K);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to set kernel arguments: " << err << "\n";
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to execute kernel: " << err << "\n";
            return err;
        }

        size_t size_C = M * N * sizeof(float);

        err = clEnqueueReadBuffer(CL::command_queue, *dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to read output array: " << err << "\n";
            return err;
        }

        return CL_SUCCESS;
    }

    cl_int BatchedMultiply(
        Kernel& kernel,
        const cl_mem* dmA, const cl_mem* dmB, cl_mem* dmC,
        float* C,
        const int M, const int N, const int K,
        const size_t batch_size
    ) {
        cl_int err;

        err = kernel.SetArgument<const cl_mem>(0, *dmA);
        err |= kernel.SetArgument<const cl_mem>(1, *dmB);
        err |= kernel.SetArgument<cl_mem>(2, *dmC);
        err |= kernel.SetArgument<const int>(3, M);
        err |= kernel.SetArgument<const int>(4, N);
        err |= kernel.SetArgument<const int>(5, K);
        err |= kernel.SetArgument<const int>(6, (int) batch_size);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to set kernel arguments: " << err << "\n";
            return err;
        }

        size_t gws[3] = { global_work_size[0], global_work_size[1], batch_size };
        size_t lws[3] = { local_work_size[0], local_work_size[1], 1 };

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 3, nullptr, gws, lws, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to execute kernel: " << err << "\n";
            return err;
        }

        size_t size_C = batch_size * M * N * sizeof(float);

        err = clEnqueueReadBuffer(CL::command_queue, *dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to read output array: " << err << "\n";
            return err;
        }

        return CL_SUCCESS;
    }

    cl_int Add(
        Kernel& kernel,
        const cl_mem* dmA, const cl_mem* dmB, cl_mem* dmC,
        float* C,
        const int size
    ) {
        cl_int err;

        err = kernel.SetArgument<const cl_mem>(0, *dmA);
        err |= kernel.SetArgument<const cl_mem>(1, *dmB);
        err |= kernel.SetArgument<cl_mem>(2, *dmC);
        err |= kernel.SetArgument<const int>(3, size);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to set kernel arguments: " << err << "\n";
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to execute kernel: " << err << "\n";
            return err;
        }

        size_t size_C = size * sizeof(float);

        err = clEnqueueReadBuffer(CL::command_queue, *dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to read output array: " << err << "\n";
            return err;
        }

        return CL_SUCCESS;
    }
}