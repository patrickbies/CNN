#include "Layer.hpp"

class DenseLayer : public Layer {
private: 
	Tensor weights;
	Tensor biases;
	size_t output_size;
	size_t input_size;

public: 
	DenseLayer(size_t output_size) : Layer(), biases({ output_size }), output_size(output_size) {}

	void initialize() override {
		std::vector<size_t> input_shape = input->getShape();	
		input_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());
		input->reshape({ input_size });

		weights = Tensor({ input_size });
		output = new Tensor({ output_size });
	}

	void forward() override {
		
	}
};