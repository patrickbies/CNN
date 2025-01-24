#pragma once

#include "Tensor.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>

class MNISTToTensor {
public:
    // Parse MNIST CSV file and return a pair of tensors (data, labels)
    static std::pair<Tensor, Tensor> parseCSV(const char* filename) {
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            throw std::runtime_error("Failed to open the file.");
        }

        std::vector<std::vector<float>> rows;
        std::string line;

        // Parse rows from the file
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> row;

            while (std::getline(ss, token, ',')) {
                try {
                    row.push_back(static_cast<float>(std::stoi(token)));
                }
                catch (const std::invalid_argument& e) {
                    throw std::runtime_error("Invalid value in CSV: " + token);
                }
                catch (const std::out_of_range& e) {
                    throw std::runtime_error("Value out of range in CSV: " + token);
                }
            }

            rows.push_back(row);
        }

        fin.close();

        if (rows.empty()) {
            throw std::runtime_error("The CSV file is empty.");
        }

        size_t num_samples = rows.size();
        size_t input_size = 28 * 28;

        // Initialize tensors for data and labels
        Tensor data({ num_samples, 1, 28, 28 }, 0.0f);
        Tensor labels({ num_samples, 10 }, 0.0f);

        for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
            const auto& row = rows[row_index];

            // Validate row size
            if (row.size() != input_size + 1) {
                throw std::runtime_error("Row " + std::to_string(row_index) + " size (" +
                    std::to_string(row.size()) + ") does not match expected size (" +
                    std::to_string(input_size + 1) + ").");
            }

            // Extract label and normalize input data
            int label = static_cast<int>(row[0]);
            if (label < 0 || label >= 10) {
                throw std::runtime_error("Invalid label value " + std::to_string(label) +
                    " at row " + std::to_string(row_index) + ".");
            }

            labels({ row_index, static_cast<size_t>(label) }) = 1.0f;

            for (size_t input_idx = 0; input_idx < input_size; ++input_idx) {
                size_t r = input_idx / 28;
                size_t c = input_idx % 28;
                data({ row_index, 0, r, c }) = row[input_idx + 1] / 255.0f;
            }
        }

        return { data, labels };
    }
};
