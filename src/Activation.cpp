#include <cmath>
#include <iostream>

#include "CL/cl.h"

#include "Activation.h"
#include "Thing.h"

namespace Activation {
    cl_int ReLU(Kernel& kernel, cl_mem& mat, const size_t len) {
        cl_int err = kernel.SetArgument<cl_mem>(0, mat);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::ReLU)\n";
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel.clkernel, 1, nullptr, &len, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::ReLU)\n";
            return err;
        }

        return CL_SUCCESS;
    }

    void Softmax(const float* inp, float* out, const int C, const int batches) { // done on the CPU
        for (int b = 0; b < batches; b++) {
            float sum = 0.0f;

            for (int i = 0; i < C; i++) {
                float exp = std::exp(inp[i + b * C]);
                sum += exp;
                out[i] = exp;
            }

            if (!sum) sum = 1e-6;

            for (int i = 0; i < C; i++) {
                out[i] /= sum;
            }
        }
    }
}