#include <array>
#include <cassert>
#include <string>

#include "Config.h"
#include "DataParser.h"
#include "FileInput.h"

#include <iostream>

namespace DataParser {
    void ParseBatch(const size_t start_line, const char* filepath, std::array<ImageData, BATCH_SIZE>& output) {
        for (size_t i = 0; i < BATCH_SIZE; i++) {
            size_t line = (start_line + i - 1) % TRAINING_ROWS + 1;

            std::string raw_text = FileInput::ReadLine(line, filepath);
            std::string current_val = "";

            size_t column = 0;

            for (size_t j = 0; j < raw_text.length();) {
                char c = raw_text.at(j++);

                while ((std::isdigit(c)/* || c == '.'*/) && j < raw_text.length()) {
                    current_val += c;
                    c = raw_text.at(j++);
                }

                if (current_val != "") {
                    if (column) output[i].pixels[column - 1] = static_cast<float>(std::stoi(current_val)) * (1 / 255.0f);
                    else output[i].digit = std::stoi(current_val);

                    current_val = "";
                    column++;
                }
            }
        }
    }
}