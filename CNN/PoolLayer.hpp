#include <cfloat>
#include "Layer.hpp"

class PoolLayer : public Layer {
private:
    size_t window_size;
    size_t stride;

public:
    PoolLayer(size_t window_size, size_t stride = 1)
        : Layer(), window_size(window_size), stride(stride) {}

    void initialize() override {
        const std::vector<size_t> input_shape = input->getShape();

        size_t output_height = (input_shape[2] - window_size) / stride + 1;
        size_t output_width = (input_shape[3] - window_size) / stride + 1;

        output = new Tensor({ input_shape[0], input_shape[1], output_height, output_width });
    }

    void forward() override {
        const std::vector<size_t> input_shape = input->getShape();
        const std::vector<size_t> output_shape = output->getShape();

        for (size_t b = 0; b < input_shape[0]; b++) {
            for (size_t c = 0; c < input_shape[1]; c++) {
                for (size_t h = 0; h < output_shape[2]; h++) {
                    for (size_t w = 0; w < output_shape[3]; w++) {
                        float mx = -FLT_MAX;

                        size_t h_start = h * stride;
                        size_t w_start = w * stride;

                        for (size_t y = 0; y < window_size; y++) {
                            for (size_t x = 0; x < window_size; x++) {
                                size_t input_h = h_start + y;
                                size_t input_w = w_start + x;

                                if (input_h < input_shape[2] && input_w < input_shape[3]) {
                                    mx = std::max(mx, (*input)({ b, c, input_h, input_w }));
                                }
                            }
                        }

                        (*output)({ b, c, h, w }) = mx;
                    }
                }
            }
        }
    }
};
