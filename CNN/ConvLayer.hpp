#include "Layer.hpp"

class ConvLayer : public Layer {
private:
	Tensor filters;
	Tensor biases;
	size_t num_filters;
	size_t filter_width;
	size_t filter_height;
	size_t stride;
	size_t padding;

	Tensor applyPadding() const {
		if (!padding) return *input;
		const std::vector<size_t> input_shape = input->getShape();
		
		Tensor padded_input({ input_shape[0], input_shape[1], input_shape[2] + 2 * padding, input_shape[3] + 2 * padding });
		for (size_t b = 0; b < input_shape[0]; b++) {
			for (size_t c = 0; c < input_shape[1]; c++) {
				for (size_t h = 0; h < input_shape[2]; h++) {
					for (size_t w = 0; w < input_shape[3]; w++) {
						padded_input({ b, c, h + padding, w + padding }) = (*input)({b, c, h, w});
					}
				}
			}
		}

		return padded_input;
	}

public:
	ConvLayer(size_t num_filters, size_t channels, size_t filter_width, 
			  size_t filter_height, size_t stride = 1, size_t padding = 0) :
		Layer(),
		filters({num_filters, channels, filter_width, filter_height}),
		biases({num_filters}),
		num_filters(num_filters),
		filter_width(filter_width),
		filter_height(filter_height),
		stride(stride),
		padding(padding) {}
	
	void initialize() override {
		const std::vector<size_t> input_shape = input->getShape();

		size_t outh = (input_shape[2] + 2 * padding - filter_height) / stride + 1;
		size_t outw = (input_shape[3] + 2 * padding - filter_width) / stride + 1;

		output = new Tensor({ input_shape[0], num_filters, outh, outw });

		size_t filter_size = filters.data.size();
		size_t output_size = input_shape[0] * num_filters * outh * outw;

		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			filters.apply([filter_size](float) { return Initializer::he_init(filter_size); });
			break;

		case (ActivationFunctions::TYPES::SIGMOID):
		case(ActivationFunctions::TYPES::SOFTMAX):
			filters.apply([filter_size, output_size](float) { return Initializer::xavier_init(filter_size, output_size); });
			break;

		default:
			filters.apply([filter_size](float) { return Initializer::uniform(filter_size); });
			break;
		}
	}

	void forward() override {
		Tensor inp = applyPadding();
		const std::vector<size_t> input_shape = inp.getShape();

		size_t output_height = output->getShape()[2];
		size_t output_width = output->getShape()[3];

		for (size_t b = 0; b < input_shape[0]; b++) {
			for (size_t f = 0; f < num_filters; f++) {
				for (size_t h = 0; h < output_height; h++) {
					for (size_t w = 0; w < output_width; w++) { 
						float sum = 0.0f;

						size_t h_start = h * stride;
						size_t w_start = w * stride;

						for (size_t c = 0; c < input_shape[1]; c++) {
							for (size_t fh = 0; fh < filter_height; fh++) {
								for (size_t fw = 0; fw < filter_width; fw++) {
									sum += filters({ f, c, fh, fw }) * inp({ b, c, h_start + fh, w_start + fw });
								}
							}
						}

						(*output)({ b, f, h, w }) = sum + biases({ f });
					}
				}
			}
		}
	}

	Tensor backward(const Tensor& gradOutput) {

	}
};