#pragma once

#include <iostream>

#include "CL/cl.h"

#include "Kernel.h"
#include "Thing.h"

// Namespace containing utility functions for handling host and device buffers
namespace Matrix {
    /**
     * Allocate device memory for a matrix, returns error code (0 means sucess)
     * host_mat - host array
     * dev_mem - device memory to store the elements into
     * size - size of the matrix (rows * columns)
     */
    template<typename T>
    cl_int Create(const T* host_mat, cl_mem& dev_mem, const size_t size) {
        cl_int err;

        cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
        if (host_mat) mem_flags |= CL_MEM_COPY_HOST_PTR;

        dev_mem = clCreateBuffer(CL::context, mem_flags, size * sizeof(T), (void*) host_mat, &err);

        if (err != CL_SUCCESS || !dev_mem) {
            ERROR_CL("Failed to allocate device memory", err);
        }

        return err;
    }

    /**
     * Calls clReleaseMemObject on the passed device memory.
     */
    void Destroy(cl_mem& dev_mem);

    /**
     * Performs C = AB, returns error code (0 means sucess)
     * kernel - matrix multiplication kernel
     * dmA - device memory for matrix A
     * dmB - device memory for matrix B
     * dmC - device memory for matrix C
     * M - number of rows in matrix A
     * N - number columns in matrix B
     * K - number of columns in matrix A, and rows in matrix B
     * gws - global work size
     * lws - local work size
     */
    cl_int Multiply(
        Kernel& kernel,
        const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
        /*float* C,*/
        const int M, const int N, const int K,
        const size_t* gws, const size_t* lws
    );

    /**
     * Performs C = AB, returns error code (0 means sucess)
     * dmA - device memory for matrix A
     * dmB - device memory for matrix B
     * dmC - device memory for matrix C
     * C - host array for matrix C
     * M - number of rows in matrix A
     * N - number columns in matrix B
     * K - number of columns in matrix A, and rows in matrix B
     * batch_size - number of batches
     */
    //cl_int BatchedMultiply(Kernel& kernel, const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC, /*float* C,*/ const int M, const int N, const int K, const int batch_size);

    /**
     * Performs C = A + B, returns error code (0 means sucess)
     * dmA - device memory for matrix A
     * dmB - device memory for matrix B
     * dmC - device memory for matrix C
     * size - size of the matrices (rows * columns)
     */
    cl_int Add(
        Kernel& kernel,
        const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
        /*float* C,*/
        const size_t size
    );

    /**
     * Multiply the matrix by a scalar value, returns error code (0 means success)
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * k - scalar multiplier
     */
    cl_int Scale(Kernel& kernel, cl_mem& mat, const size_t size, const float k);

    /**
     * Copy device memory buffer into host memory buffer, returns error code (0 means success)
     * src - device memory buffer
     * dest - host memory buffer
     * size - number of elements to copy
     */
    template<typename T>
    cl_int Transfer(cl_mem& src, T* dest, size_t size) {
        cl_int err = clEnqueueReadBuffer(CL::command_queue, src, CL_TRUE, 0, size * sizeof(T), (void*) dest, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to read into host buffer", err);
        }

        return err;
    }

    /**
     * Copy host memory buffer into device memory buffer, returns error code (0 means success)
     * src - host memory buffer
     * dest - device memory buffer
     * size - number of elements to copy
     */
    template<typename T>
    cl_int Transfer(T* src, cl_mem& dest, size_t size) {
        cl_int err = clEnqueueWriteBuffer(CL::command_queue, dest, CL_TRUE, 0, size * sizeof(T), (void*) src, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            ERROR_CL("Failed to write into device buffer", err);
        }

        return err;
    }

    /**
     * Sets all values in the matrix to the specified number, returns error code (0 means success)
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * value - value to populate the matrix with
     */
    cl_int Populate(
        Kernel& kernel,
        cl_mem& mat,
        const size_t size, const float value
    );

    /**
     * Randomise each element in a matrix with a uniform distribution of floats in the range [min, max)
     * Optionally provide a random seed
     * Returns error code (0 means success)
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * min - lower bound
     * max - upper bound
     * seed (optional) - random seed
     */
    cl_int Randomise(
        Kernel& kernel,
        cl_mem& mat,
        const size_t size, const float min, const float max, uint32_t seed = 0
    );
}