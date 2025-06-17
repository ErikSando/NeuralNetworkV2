#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "Config.h"
#include "FileInput.h"

namespace FileInput {
    static std::array<uint32_t, TRAINING_ROWS> line_offsets;

    std::string lastpath;
    //static std::streampos jump = 0;
    size_t last_line = 0;
    size_t largest_line = 1;

    std::string ReadLine(const size_t line, const std::string& filepath) {
        std::ifstream input(filepath);

        if (!input.is_open()) {
            std::cout << "Failed to open file: " << filepath << "\n";
            return "";
        }

        if (filepath != lastpath) {
            last_line = 0;
            largest_line = 1;
            lastpath = filepath;
        }

        size_t current_line = line;

        if (line > largest_line) {
            assert(line >= 1 && line <= TRAINING_ROWS);
            assert(largest_line >= 1 && largest_line <= TRAINING_ROWS);

            current_line = largest_line;
        }

        std::streampos jump = static_cast<std::streampos>(line_offsets[current_line - 1]);
        input.seekg(jump);

        std::string line_string;

        for (; current_line <= line; current_line++) {
            if (!std::getline(input, line_string)) {
                throw std::out_of_range("Requested line number out of range");
            }

            std::streampos pos = input.tellg();

            if (pos != EOF) {
                //jump = pos;
                line_offsets[current_line] = static_cast<uint32_t>(pos);

                if (current_line > largest_line) {
                    largest_line = current_line;
                }
            }
        }

        last_line = line;

        return line_string;
    }
}