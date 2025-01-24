#pragma once

#include "Layer.hpp"

class DenseLayer : public Layer {
private: 
	size_t output_size;
	size_t input_size = 0;
	size_t num_batches = 0;

	std::vector<size_t> flatBatchIndex(size_t b, size_t ind) {
		const std::vector<size_t>& shape = input->getShape();
		if (shape.size() < 2) {
			throw std::invalid_argument("Tensor shape must have at least two dimensions (batch dimension and more).");
		}

		size_t numDimensions = shape.size();
		std::vector<size_t> coordinates(numDimensions, 0);

		coordinates[0] = b;

		size_t remainingIndex = ind;
		for (size_t dim = numDimensions - 1; dim > 0; --dim) {
			size_t stride = shape[dim];
			coordinates[dim] = remainingIndex % stride; 
			remainingIndex /= stride;                  
		}

		if (remainingIndex != 0) {
			throw std::out_of_range("Flat index exceeds tensor dimensions.");
		}

		return coordinates;
	}

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
					sum += (*input)(flatBatchIndex(b, j)) * weights({ j, i });
				}
				(*output)({ b, i }) = sum;
			}
		}
	}

	void backward(const Tensor& gradOutput) override {
		for (size_t b = 0; b < num_batches; b++) {
			for (size_t i = 0; i < output_size; i++) {
				(*bias_gradient)({ i }) += gradOutput({ b, i });
				for (size_t j = 0; j < input_size; j++) {
					(*weight_gradient)({ j, i }) += (*input)(flatBatchIndex(b, j)) * gradOutput({ b, i });
				}
			}
		}

		for (size_t b = 0; b < num_batches; b++) {
			for (size_t j = 0; j < input_size; j++) {
				float sum = 0.0f;
				for (size_t i = 0; i < output_size; i++) {
					sum += weights({ j, i }) * gradOutput({ b, i });
				}
				(*input_gradient)(flatBatchIndex(b, j)) = sum;
			}
		}
	}
};