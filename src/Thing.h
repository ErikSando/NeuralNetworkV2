#pragma once

#include <CL/cl.h>

#define GPU 1

namespace CL {
    extern cl_platform_id platform_ids[100];
    extern cl_device_id device_id;
    extern cl_context context;
    extern cl_command_queue command_queue;

    extern bool setup_success;

    extern void Init();
}