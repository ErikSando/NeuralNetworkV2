#pragma once

#include "CL/cl.h"

constexpr std::string_view FILE_NAME(std::string_view path) {
    size_t last_slash = path.find_last_of("/\\");
    return last_slash == std::string_view::npos ? path : path.substr(last_slash + 1);
}

namespace CL {
    extern cl_platform_id platform_ids[100];
    extern cl_device_id device_id;
    extern cl_ulong global_mem_size;
    extern cl_context context;
    extern cl_command_queue command_queue;

    extern cl_int Init();
    extern void Destroy();
}

extern int CommandLoop();