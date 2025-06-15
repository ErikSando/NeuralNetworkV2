#pragma once

#include <array>

#include "Config.h"

struct ImageData {
    std::array<float, NODE_COUNT[INPUT]> pixels;
    int digit;
};

namespace DataParser {
    void ParseBatch(const size_t start_line, const char* filepath, std::array<ImageData, BATCH_SIZE>& output);
}