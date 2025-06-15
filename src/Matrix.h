#pragma once

#include "CL/cl.h"

#include "Kernel.h"

namespace Matrix {
    /**
     * Allocate device memory for a matrix, returns error code (0 means sucess)
     * host_mat - host array
     * dev_mem - device memory to store the elements into
     * size - size of the matrix (rows * columns)
     */
    cl_int Create(const float* host_mat, cl_mem& dev_mem, const size_t size);

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
        Kernel* kernel,
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
        Kernel* kernel,
        const cl_mem& dmA, const cl_mem& dmB, cl_mem& dmC,
        /*float* C,*/
        const size_t size
    );

    /**
     * Multiply the matrix by a scalar value
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * k - scalar multiplier
     */
    cl_int Scale(Kernel* kernel, cl_mem& mat, const size_t size, const float k);

    /**
     * Copy device memory buffer into host memory buffer
     * src - device memory buffer
     * dest - host memory buffer
     * size - number of bytes to copy
     */
    cl_int Transfer(cl_mem& src, float* dest, size_t size);
    
    /**
     * Sets all values in the matrix to the specified number
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * value - value to populate the matrix with
     */
    cl_int Populate(
        Kernel* kernel,
        cl_mem& mat,
        const size_t size, const float value
    );

    /**
     * Randomise each element in a matrix with a uniform distribution of floats in the range [min, max)
     * Optionally provide a random seed
     * mat - device memory of the matrix
     * size - size of the matrix (rows * columns)
     * min - lower bound
     * max - upper bound
     * seed (optional) - random seed
     */
    cl_int Randomise(
        Kernel* kernel,
        cl_mem& mat,
        const size_t size, const float min, const float max, uint32_t seed = 0
    );
}