#pragma once

#include "Layer.hpp"

class ConvLayer : public Layer {
private:
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
						padded_input(b, c, h + padding, w + padding) = (*input)(b, c, h, w);
					}
				}
			}
		}

		return padded_input;
	}

public:
	ConvLayer(size_t num_filters, size_t filter_width, 
			  size_t filter_height, size_t stride = 1, size_t padding = 0, 
			  ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE) :
		Layer(_ac),
		num_filters(num_filters),
		filter_width(filter_width),
		filter_height(filter_height),
		stride(stride),
		padding(padding)
	{
	}
	
	void initialize(std::vector<size_t> input_shape) override {
		weights = Tensor({ num_filters, input_shape[1], filter_width, filter_height});
		biases = Tensor({ num_filters });

		size_t outh = (input_shape[2] + 2 * padding - filter_height) / stride + 1;
		size_t outw = (input_shape[3] + 2 * padding - filter_width) / stride + 1;

		output = new Tensor({ input_shape[0], num_filters, outh, outw });

		size_t filter_size = weights.data.size();
		size_t output_size = input_shape[0] * num_filters * outh * outw;

		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			Initializer::he_init(weights, filter_size);
			break;

		case (ActivationFunctions::TYPES::SIGMOID):
		case(ActivationFunctions::TYPES::SOFTMAX):
			Initializer::xavier_init(weights, filter_size, output_size);
			break;

		default:
			Initializer::uniform(weights, filter_size);
			break;
		}

		weight_gradient = new Tensor(weights.getShape());
		bias_gradient = new Tensor({ num_filters });
		input_gradient = new Tensor(input_shape);
	}

	void forward() override {
		const std::vector<size_t>& input_shape = input->getShape();

		size_t output_height = output->getShape()[2];
		size_t output_width = output->getShape()[3];

#pragma omp parallel for collapse(4)
		for (size_t b = 0; b < input_shape[0]; b++) {
			for (size_t f = 0; f < num_filters; f++) {
				for (size_t h = 0; h < output_height; h++) {
					for (size_t w = 0; w < output_width; w++) {
						float sum = 0.0f;

						size_t h_start = h * stride;
						size_t w_start = w * stride;

						for (size_t c = 0; c < input_shape[1]; c++) {
							for (size_t fh = 0; fh < filter_height; fh++) {
								size_t h_index = h_start + fh;
								if (h_index >= input_shape[2]) continue;

								for (size_t fw = 0; fw < filter_width; fw++) {
									size_t w_index = w_start + fw;
									if (w_index >= input_shape[3]) continue;

									sum += weights(f, c, fh, fw) * (*input)(b, c, h_index, w_index);
								}
							}
						}

						(*output)(b, f, h, w) = sum + biases.data[f];
					}
				}
			}
		}
	}

	void backward(const Tensor& gradOutput) override {
		std::vector<size_t> input_shape = input->getShape();
		std::vector<size_t> output_shape = output->getShape();

		input_gradient->zero();
		weight_gradient->zero();
		bias_gradient->zero();

#pragma omp parallel for collapse(6);
		for (size_t b = 0; b < input_shape[0]; b++) {
			for (size_t c = 0; c < input_shape[1]; c++) {
				for (size_t i = 0; i < input_shape[2]; i++) {
					for (size_t j = 0; j < input_shape[3]; j++) {
						for (size_t o = 0; o < num_filters; o++) {
							for (size_t p = 0; p < output_shape[2]; p++) {
								for (size_t q = 0; q < output_shape[3]; q++) {
									size_t h_start = p * stride;
									size_t w_start = q * stride;
									if (i >= h_start && i < h_start + filter_height &&
										j >= w_start && j < w_start + filter_width) {
										size_t fh = i - h_start;
										size_t fw = j - w_start;
										
										(*input_gradient)(b, c, i, j) +=
											gradOutput(b, o, p, q) * weights(o, c, fh, fw);
									}
								}
							}
						}
					}
				}
			}	
		}

		for (size_t o = 0; o < num_filters; o++) {
			for (size_t c = 0; c < input_shape[1]; c++) {
				for (size_t fh = 0; fh < filter_height; fh++) {
					for (size_t fw = 0; fw < filter_width; fw++) {
						for (size_t b = 0; b < input_shape[0]; b++) {
							for (size_t p = 0; p < output_shape[2]; p++) {
								for (size_t q = 0; q < output_shape[3]; q++) {
									(*weight_gradient)(o, c, fh, fw) +=
										gradOutput(b, o, p, q) * (*input)(b, c, p * stride + fh, q * stride + fw);
								}
							}
						}
					}
				}
			}
		}

		for (size_t o = 0; o < num_filters; o++) {
			for (size_t b = 0; b < input_shape[0]; b++) {
				for (size_t p = 0; p < output_shape[2]; p++) {
					for (size_t q = 0; q < output_shape[3]; q++) {
						bias_gradient->data[o] += gradOutput(b, o, p, q);
					}
				}
			}
		}
	}
};