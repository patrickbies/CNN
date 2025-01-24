#include "Tensor.hpp"
#include "ActivationFunctions.hpp"
#include "Initializer.hpp"

class Layer {
protected: 
	Tensor* input;
	Tensor* output;
	ActivationFunctions::TYPES activation_function;

public: 
	Layer(ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE) : input(nullptr), output(nullptr), activation_function(_ac) {};
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
	
	virtual void initialize() = 0;
	virtual void forward() = 0;
	virtual Tensor backward(const Tensor& gradOutput) = 0; // gradoutput should be size of output
};