#pragma once

#include <cfloat>
#include "Layer.hpp"

class PoolLayer : public Layer {
private:
    size_t window_size;
    size_t stride;
    Tensor max_indices;

public:
    PoolLayer(size_t window_size, size_t stride = 1, ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE)
        : Layer(_ac), window_size(window_size), stride(stride) {}

    void initialize(std::vector<size_t> is) override {
        input_shape = is;

        size_t output_height = (input_shape[2] - window_size) / stride + 1;
        size_t output_width = (input_shape[3] - window_size) / stride + 1;

        output_shape = { input_shape[0], input_shape[1], output_height, output_width };
    }

    void initOutput(size_t batches) override {
        if (!output_shape.size()) {
            throw std::exception("Layer must be intialized prior to setting the number of batches");
        }

        input_shape[0] = batches;
        output_shape[0] = batches;
        output = new Tensor(output_shape);

        input_gradient = new Tensor(input_shape);
        max_indices = Tensor(output_shape, -1);
    }

    void forward() override {
        for (size_t b = 0; b < input_shape[0]; b++) {
            for (size_t c = 0; c < input_shape[1]; c++) {
                for (size_t h = 0; h < output_shape[2]; h++) {
                    for (size_t w = 0; w < output_shape[3]; w++) {
                        float mx = -FLT_MAX;
                        size_t max_index = -1;

                        size_t h_start = h * stride;
                        size_t w_start = w * stride;

                        for (size_t y = 0; y < window_size; y++) {
                            for (size_t x = 0; x < window_size; x++) {
                                size_t input_h = h_start + y;
                                size_t input_w = w_start + x;

                                if (input_h < input_shape[2] && input_w < input_shape[3]) {
                                    float cur = (*input)(b, c, input_h, input_w);
                                    if (cur > mx) {
                                        mx = cur;
                                        max_index = input_h * input_shape[3] + input_w;
                                    }
                                }
                            }
                        }

                        max_indices(b, c, h, w) = max_index;
                        (*output)(b, c, h, w) = mx;
                    }
                }
            }
        }
    }

    void backward(const Tensor& gradOutput) override {
        for (size_t b = 0; b < output_shape[0]; b++) {
            for (size_t c = 0; c < output_shape[1]; c++) {
                for (size_t h = 0; h < output_shape[2]; h++) {
                    for (size_t w = 0; w < output_shape[3]; w++) {
                        size_t flat_index = max_indices({ b, c, h, w });
                        size_t input_h = flat_index / input_shape[3];
                        size_t input_w = flat_index % input_shape[3];
                        (*input_gradient)(b, c, input_h, input_w) += gradOutput(b, c, h, w);
                    }
                }
            }
        }
    }
};
