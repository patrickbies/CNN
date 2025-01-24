#include "Tensor.hpp"
#include <fstream>
#include <string>
#include <sstream>

class MNISTToTensor {
    // returns <data, labels> (function specifically for MNIST CSV dataset):
	static std::pair<Tensor, Tensor> parseCSV(const char* filename) {
		std::fstream fin;

		fin.open(filename, std::ios::in);

		std::vector<std::vector<float>> row;
        std::string line, word, temp;
        size_t i = 0;

        while (fin >> temp)
        {
            row.clear();

            std::getline(fin, line);
            std::stringstream s(line);
            std::string token;

            while (std::getline(s, token, ',')) {
                if (!row[i].size()) row[i].push_back(static_cast<float>(std::stoi(token)));
                else row[i].push_back(static_cast<float>(std::stoi(token)) / 255.0f); // pixel values b/w 0 and 1;
            }

            i++;
        }

        fin.close();
		
        Tensor data({row.size(), 1, 28, 28}), labels({row.size(), 10});// initialized all to 0;

        for (size_t i = 0; i < row.size(); i++) {
            labels({ i, static_cast<unsigned long long>(row[i][0]) }) = 1.0f;

            for (size_t h = 0; h < 28; h++) {
                for (size_t w = 0; w < 28; w++) {
                    data({ i, 1, h, w }) = row[i][1 + h * 28 + w];
                }
            }
        }

        return { data, labels };
	}
};