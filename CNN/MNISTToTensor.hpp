#include "Tensor.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

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

        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> row;

            while (std::getline(ss, token, ',')) {
                row.push_back(static_cast<float>(std::stoi(token)));
            }

            rows.push_back(row);
        }

        fin.close();

        if (rows.empty()) {
            throw std::runtime_error("The CSV file is empty.");
        }

        size_t num_samples = rows.size();
        size_t input_size = 28 * 28;

        Tensor data({ num_samples, 1, 28, 28 }, 0.0f);
        Tensor labels({ num_samples, 10 }, 0.0f);      

        for (size_t i = 0; i < num_samples; ++i) {
            const auto& row = rows[i];

            if (row.size() != input_size + 1) {
                throw std::runtime_error("Row size does not match expected MNIST format.");
            }

            int label = static_cast<int>(row[0]);
            if (label < 0 || label >= 10) {
                throw std::runtime_error("Invalid label value in the dataset.");
            }
            labels({ i, static_cast<size_t>(label) }) = 1.0f;

            for (size_t j = 0; j < input_size; ++j) {
                size_t row_index = j / 28;
                size_t col_index = j % 28;
                data({ i, 0, row_index, col_index }) = row[j + 1] / 255.0f;
            }
        }

        return { data, labels };
    }
};
