#include "Tensor.hpp"

class Layer {
protected: 
	Tensor* input;
	Tensor* output;

public: 
	Layer() : input(nullptr), output(nullptr) {};
	virtual ~Layer() {}

	void serInput(Tensor* _input) {
		input = _input;
	}

	Tensor* getInput() const {
		return input;
	}

	Tensor* getOutput() const {
		return output;
	}
	
	virtual void initialize() = 0;
	virtual Tensor forward(const Tensor& input) = 0;
	virtual Tensor backward(const Tensor& gradOutput) { // optional
		throw std::runtime_error("n/a for this layer.");
	}
};