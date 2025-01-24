#pragma once

#include "Layer.hpp"

class ActivationLayer : public Layer {
public:
	ActivationLayer(ActivationFunctions::TYPES _activation_function) : Layer(_activation_function) {}

	void initialize() override {}

	void forward() override {
		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			(*output) = ActivationFunctions::relu(*input);
			break;
		case (ActivationFunctions::TYPES::SIGMOID):
			(*output) = ActivationFunctions::sigmoid(*input);
			break;
		case (ActivationFunctions::TYPES::SOFTMAX_CEL):
		case (ActivationFunctions::TYPES::SOFTMAX):
			(*output) = ActivationFunctions::softmax(*input);
			break;
		default:
			throw std::invalid_argument("Unsupported activation function.");
		}
	}

	Tensor backward(const Tensor& gradOutput) override {
		Tensor res = gradOutput;

		switch (activation_function) {
		case (ActivationFunctions::TYPES::RELU):
			res = gradOutput * ActivationFunctions::relu_derivative(*input);
			break;
		case (ActivationFunctions::TYPES::SIGMOID):
			res = gradOutput * ActivationFunctions::sigmoid_derivative(*input);
			break;
		case (ActivationFunctions::TYPES::SOFTMAX):
			res = gradOutput * ActivationFunctions::softmax_derivative(*input);
			break;
		case (ActivationFunctions::TYPES::SOFTMAX_CEL):
			break;
		default:
			throw std::invalid_argument("Unsupported activation function.");
		}

		return res;
	}
};