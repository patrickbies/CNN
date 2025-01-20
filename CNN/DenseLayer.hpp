#include "Layer.hpp"

class DenseLayer : public Layer {
private: 
	Tensor weights;
	Tensor biases;
	size_t output_size;
	size_t input_size = 0;
	size_t num_batches = 0;

	std::vector<size_t> flatBatchIndex(size_t b, size_t ind) {
		std::vector<size_t> s = input->getShape();

		size_t width = s[3];
		size_t height = s[2];
		size_t channels = s[1];

		size_t c = ind / (height * width);     
		size_t hw_index = ind % (height * width);
		size_t h = hw_index / width;            
		size_t w = hw_index % width;             

		return { b, c, h, w };
	}

public: 
	DenseLayer(size_t output_size) : Layer(), output_size(output_size) {}

	void initialize() override {
		std::vector<size_t> input_shape = input->getShape();	

		num_batches = input_shape[0];
		input_size = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<>());

		weights = Tensor({ input_size, output_size });
		biases = Tensor({ output_size });
		output = new Tensor({ num_batches, output_size });

		size_t _is = input_size, _os = output_size;

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
};