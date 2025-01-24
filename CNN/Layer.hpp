#pragma once

#include "Tensor.hpp"
#include "ActivationFunctions.hpp"
#include "Initializer.hpp"

class Layer {
protected: 
	Tensor* input = nullptr;
	Tensor* output = nullptr;
	Tensor* input_gradient = nullptr;
	Tensor* weight_gradient = nullptr;
	Tensor* bias_gradient = nullptr;
	ActivationFunctions::TYPES activation_function;

public: 
	Tensor biases;
	Tensor weights;

	Layer(ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE) : activation_function(_ac) {};
	virtual ~Layer() {}

	void setInput(Tensor* _input) {
		input = _input;
	}

	void setActivationFunction(ActivationFunctions::TYPES function) {
		activation_function = function;
	};

	ActivationFunctions::TYPES getActivationFunction() const {
		return activation_function;
	}

	Tensor* getInput() const {
		return input;
	}

	Tensor* getOutput() const {
		return output;
	}

	Tensor* getInputGradient() const {
		return input_gradient;
	}

	Tensor* getWeightGradient() const {
		return weight_gradient;
	}

	Tensor* getBiasGradient() const {
		return bias_gradient;
	}

	virtual void initialize(std::vector<size_t> input_shape) = 0;
	virtual void forward() = 0;

	// returns tuple of <input gradient, weight gradient, bias gradient> wrt loss:
	virtual void backward(const Tensor& gradOutput) = 0; // gradoutput should be size of output
};