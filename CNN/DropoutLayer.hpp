#include "Layer.hpp"

class DropoutLayer : public Layer {
private:
	float p;
public:
	DropoutLayer(float p) : Layer(), p(p) {}

	void initialize(std::vector<size_t> input_shape) {};
	void forward() {};
	void backward(const Tensor& gradOutput) {}
};