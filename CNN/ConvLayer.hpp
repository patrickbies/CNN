#include "Layer.hpp"

class ConvLayer : public Layer {
public: 
	int numFilters;
	int kernalSize;
	int stride;
	int padding;
	std::vector<Tensor> filters;
	Tensor biases;
	Tensor filters;

	Tensor forward(const Tensor& input);
	Tensor backward(const Tensor& gradOutput);
	void updateWeights(float learningRate);
};