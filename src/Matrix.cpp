#include <iostream>

#include "CL/cl.h"

#include "Kernel.h"
#include "Matrix.h"
#include "Thing.h"

namespace Matrix {
    // cl_int Create(const float* host_mat, cl_mem& dev_mem, const size_t size) {
    //     cl_int err;

    //     cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
    //     if (host_mat) mem_flags |= CL_MEM_COPY_HOST_PTR;

    //     dev_mem = clCreateBuffer(CL::context, mem_flags, size * sizeof(float), (void*) host_mat, &err);

    //     if (err != CL_SUCCESS || !dev_mem) {
    //         std::cout << "Failed to allocate device memory: " << err << " (" << FILE_NAME(__FILE__) << " > Matrix::Create)\n";
    //     }

    //     return err;
    // }

    void Destroy(cl_mem& dev_mem) {
        if (dev_mem) clReleaseMemObject(dev_mem);
    }

    cl_int Multiply(
        Kernel& kernel,
        const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
        //float* C,
        const int M, const int N, const int K,
        const size_t* gws, const size_t* lws
    ) {
        cl_int err;

        err = kernel.SetArgument(0, dmA);
        err |= kernel.SetArgument(1, dmB);
        err |= kernel.SetArgument(2, dmC);
        err |= kernel.SetArgument(3, M);
        err |= kernel.SetArgument(4, N);
        err |= kernel.SetArgument(5, K);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to set kernel arguments", err);
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 2, nullptr, gws, lws, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to execute kernel", err);
            // return err;
        }

        // size_t size_C = M * N * sizeof(float);

        // err = clEnqueueReadBuffer(CL::command_queue, dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

        // if (err != CL_SUCCESS) {
        //     std::cout << "Failed to read output array: " << err << "\n";
        // }

        return err;
    }

    // cl_int BatchedMultiply(
    //     Kernel* kernel,
    //     const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
    //     //float* C,
    //     const int M, const int N, const int K,
    //     const int batch_size
    // ) {
    //     cl_int err;

    //     err = kernel->SetArgument(0, dmA);
    //     err |= kernel->SetArgument(1, dmB);
    //     err |= kernel->SetArgument(2, dmC);
    //     err |= kernel->SetArgument(3, M);
    //     err |= kernel->SetArgument(4, N);
    //     err |= kernel->SetArgument(5, K);
    //     err |= kernel->SetArgument(6, batch_size);

    //     if (err != CL_SUCCESS) {
    //         std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > Matrix:BatchedMultiply)\n";
    //         return err;
    //     }

    //     size_t gws[3] = { global_work_size[0], global_work_size[1], (size_t) batch_size };
    //     size_t lws[3] = { local_work_size[0], local_work_size[1], 1 };

    //     err = clEnqueueNDRangeKernel(CL::command_queue, kernel->clkernel, 3, nullptr, gws, lws, 0, nullptr, nullptr);

    //     if (err != CL_SUCCESS) {
    //         std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > Matrix:BatchedMultiply)\n";
    //         // return err;
    //     }

    //     // size_t size_C = batch_size * M * N * sizeof(float);

    //     // err = clEnqueueReadBuffer(CL::command_queue, dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

    //     // if (err != CL_SUCCESS) {
    //     //     std::cout << "Failed to read output array: " << err << "\n";
    //     // }

    //     return err;
    // }

    cl_int Add(
        Kernel& kernel,
        const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
        //float* C,
        const size_t size
    ) {
        cl_int err;

        err = kernel.SetArgument(0, dmA);
        err |= kernel.SetArgument(1, dmB);
        err |= kernel.SetArgument(2, dmC);
        // err |= kernel.SetArgument(3, size);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to set kernel arguments", err);
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to execute kernel", err);
            // return err;
        }

        // size_t size_C = size * sizeof(float);

        // err = clEnqueueReadBuffer(CL::command_queue, dmC, CL_TRUE, 0, size_C, (void*) C, 0, nullptr, nullptr);

        // if (err != CL_SUCCESS) {
        //     std::cout << "Failed to read output array: " << err << "\n";
        // }

        return err;
    }

    cl_int Scale(Kernel& kernel, cl_mem& mat, const size_t size, const float k) {
        cl_int err;

        err = kernel.SetArgument(0, mat);
        err |= kernel.SetArgument(1, k);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to set kernel arguments", err);
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to execute kernel", err);
        }

        return err;
    }

    // cl_int Transfer(cl_mem& src, float* dest, size_t size) {
    //     cl_int err = clEnqueueReadBuffer(CL::command_queue, src, CL_TRUE, 0, size, (void*) dest, 0, nullptr, nullptr);

    //     if (err != CL_SUCCESS) {
    //         std::cout << "Failed to read output array: " << err << " (" << FILE_NAME(__FILE__) << " > Matrix::Transfer)\n";
    //     }

    //     return err;
    // }

    cl_int Populate(Kernel& kernel, cl_mem& mat, const size_t size, const float value) {
        cl_int err;

        err = kernel.SetArgument(0, mat);
        err |= kernel.SetArgument(1, value);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to set kernel arguments", err);
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to execute kernel", err);
        }

        return err;
    }

    cl_int Randomise(Kernel& kernel, cl_mem& mat, const size_t size, const float min, const float max, uint32_t seed) {
        cl_int err;

        if (!seed) {
            seed = rand();
        }

        err = kernel.SetArgument(0, mat);
        err |= kernel.SetArgument(1, min);
        err |= kernel.SetArgument(2, max);
        err |= kernel.SetArgument(3, seed);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to set kernel arguments", err);
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to execute kernel", err);
        }

        return err;
    }
}