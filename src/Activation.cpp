#include <cmath>
#include <iostream>

#include "CL/cl.h"

#include "Activation.h"
#include "Thing.h"

namespace Activation {
    cl_int ReLU(Kernel* kernel, cl_mem& mem, const size_t size) {
        cl_int err = kernel->SetArgument<cl_mem>(0, mem);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::ReLU)\n";
            return err;
        }

        err = clEnqueueNDRangeKernel(CL::command_queue, kernel->clkernel, 1, nullptr, &size, nullptr, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::ReLU)\n";
            return err;
        }

        return CL_SUCCESS;
    }

    cl_int Softmax(Kernel* kernel, cl_mem& mem, const size_t C, const int batches) {
        // probabaly could utilise the GPU better here
        for (int b = 0; b < batches; b++) {
            cl_int err;
            err = kernel->SetArgument<cl_mem>(0, mem);
            err |= kernel->SetArgument<int>(1, b * C);

            if (err != CL_SUCCESS) {
                std::cout << "Failed to set kernel arguments: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::Softmax)\n";
                return err;
            }

            err = clEnqueueNDRangeKernel(CL::command_queue, kernel->clkernel, 1, nullptr, &C, nullptr, 0, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                std::cout << "Failed to execute kernel: " << err << " (" << FILE_NAME(__FILE__) << " > Activation::Softmax)\n";
                return err;
            }
        }

        return CL_SUCCESS;
    }

    void Softmax(float* mem, const int C, const int batches) {
        for (int b = 0; b < batches; b++) {
            float sum = 0.0f;

            for (int i = 0; i < C; i++) {
                float exp = std::exp(mem[i + b * C]);
                sum += exp;
                mem[i + b * C] = exp;
            }

            if (!sum) sum = 1e-6;

            for (int i = 0; i < C; i++) {
                mem[i + b * C] /= sum;
            }
        }
    }
}