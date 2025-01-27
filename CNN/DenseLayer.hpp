#pragma once

#include "Layer.hpp"

class DenseLayer : public Layer {
private: 
	size_t output_size;
	size_t input_size = 0;
	size_t num_batches = 0;

public: 
	DenseLayer(size_t output_size, ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE) : Layer(_ac), output_size(output_size) {}

	void initialize(std::vector<size_t> input_shape) override {
		num_batches = input_shape[0];
		input_size = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<>());

		size_t _is = input_size;
		size_t _os = output_size;

		weights = Tensor({ input_size, output_size });
		biases = Tensor({ output_size });

		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			weights.apply([_is](float) { return Initializer::he_init(_is); });
			break;

		case (ActivationFunctions::TYPES::SIGMOID):
		case(ActivationFunctions::TYPES::SOFTMAX):
			weights.apply([_is, _os](float) { return Initializer::xavier_init(_is, _os); });
			break;

		default:
			weights.apply([_is](float) { return Initializer::uniform(_is); });
			break;
		}

		weight_gradient = new Tensor({ input_size, output_size });
		bias_gradient = new Tensor({ output_size });
		input_gradient = new Tensor(input_shape);
		output = new Tensor({ num_batches, output_size });
	}

	void forward() override {
		for (size_t b = 0; b < num_batches; b++) {
			for (size_t i = 0; i < output_size; i++) {
				float sum = biases({ i });
				for (size_t j = 0; j < input_size; j++) {
					sum += (*input)({ b, j }) * weights({ j, i });
				}
				(*output)({ b, i }) = sum;
			}
		}
	}

	void backward(const Tensor& gradOutput) override {
		input_gradient->zero();
		weight_gradient->zero();
		bias_gradient->zero();

		for (size_t b = 0; b < num_batches; b++) {
			for (size_t i = 0; i < output_size; i++) {
				(*bias_gradient)({ i }) += gradOutput({ b, i });
				for (size_t j = 0; j < input_size; j++) {
					(*weight_gradient)({ j, i }) += (*input)({ b, j }) * gradOutput({ b, i });
					(*input_gradient)({ b, j }) += weights({ j, i }) * gradOutput({ b, i });
				}
			}
		}
	}
};