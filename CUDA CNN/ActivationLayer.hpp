#pragma once

#include "Layer.hpp"

class ActivationLayer : public Layer {
public:
	ActivationLayer(ActivationFunctions::TYPES _activation_function) : Layer(_activation_function) {}

	void initialize(std::vector<size_t> is) override {
		input_shape = is;
		output_shape = is;
	}

	void forward() override {
		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			ActivationFunctions::relu(*output, *input);
			break;
		case (ActivationFunctions::TYPES::SIGMOID):
			ActivationFunctions::sigmoid(*output, *input);
			break;
		case (ActivationFunctions::TYPES::SOFTMAX_CEL):
		case (ActivationFunctions::TYPES::SOFTMAX):
			ActivationFunctions::softmax(*output, *input);
			break;
		default:
			throw std::invalid_argument("Unsupported activation function.");
		}
	}

	void backward(const Tensor& gradOutput) override {
		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			ActivationFunctions::relu_derivative(*input_gradient, *input);
			*input_gradient = gradOutput * *input_gradient; 
			break;
		case (ActivationFunctions::TYPES::SIGMOID):
			ActivationFunctions::sigmoid_derivative(*input_gradient, *input);
			*input_gradient = gradOutput * *input_gradient;
			break;
		case (ActivationFunctions::TYPES::SOFTMAX):
			ActivationFunctions::softmax_derivative(*input_gradient, *input);
			*input_gradient = gradOutput * *input_gradient;
			break;
		case (ActivationFunctions::TYPES::SOFTMAX_CEL):
			*input_gradient = gradOutput;
			break;
		default:
			throw std::invalid_argument("Unsupported activation function.");
		}
	}
};