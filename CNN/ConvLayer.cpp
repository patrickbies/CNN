#include "ConvLayer.hpp"

Tensor ConvLayer::forward(const Tensor& input) 
{
	std::vector<size_t> output_size;
	// output size is x by y by numfilters by d
	Tensor output(output_size);
}

Tensor ConvLayer::backward(const Tensor& gradOutput) 
{

}

void ConvLayer::updateWeights(float learningRate)
{

}