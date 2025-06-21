#include <iostream>

#include "Config.h"
#include "Thing.h"

int main() {
    cl_int err = CL::Init();

    if (err != CL_SUCCESS) {
        std::cout << "Failed to intialise OpenCL (" << FILE_NAME(__FILE__) << ")\n";
        return 1;
    }

    std::cout << "Batch size: " << BATCH_SIZE << " samples\n";

    int _ = CommandLoop();

    CL::Destroy();

    return _;
}
