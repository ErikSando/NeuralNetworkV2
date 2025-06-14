#pragma once

#include <CL/cl.h>

#include "Kernel.h"

namespace Matrix {
    /**
     * Allocate device memory for a matrix, returns error code (0 means sucess)
     * mat - array for the matrix
     * size - length of the matrix (rows * columns)
     */
    cl_int Create(const float* mat, cl_mem* dev_mem, const int size);

    /**
     * Calls clReleaseMemObject on the passed device memory.
     */
    void Destroy(cl_mem& dev_mem);

    /**
     * Performs C = AB, returns error code (0 means sucess)
     * dmA - device memory for matrix A
     * dmB - device memory for matrix B
     * dmC - device memory for matrix C
     * C - array for matrix C
     * rA - number of rows in matrix A
     * cB - number columns in matrix B
     */
    cl_int Multiply(Kernel& kernel, const cl_mem* dmA, const cl_mem* dmB, cl_mem* dmC, float* C, const int cA, const int cB);

    /**
     * Performs C = A + B, returns error code (0 means sucess)
     * dmA - device memory for matrix A
     * dmB - device memory for matrix B
     * dmC - device memory for matrix C
     * C - array for matrix C
     * size - length of the matrices (rows * columns)
     */
    cl_int Add(Kernel& kernel, const cl_mem* dmA, const cl_mem* dmB, cl_mem* dmC, float* C, const int size);
}