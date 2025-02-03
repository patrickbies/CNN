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
	std::vector<size_t> input_shape;
	std::vector<size_t> output_shape;

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

	std::vector<size_t> getOutputShape() const {
		return output_shape;
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

	virtual void initOutput(size_t batches) {
		if (!output_shape.size()) {
			throw std::exception("Layer must be intialized prior to setting the number of batches");
		}

		input_shape[0] = batches;
		output_shape[0] = batches;
		output = new Tensor(output_shape);

		input_gradient = new Tensor(input_shape);
	};
	virtual void initialize(std::vector<size_t> input_shape) = 0;
	virtual void forward() = 0;
	virtual void backward(const Tensor& gradOutput) = 0;
};