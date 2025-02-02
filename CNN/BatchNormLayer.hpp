#include "Layer.hpp"

class BatchNormLayer : public Layer {
private:

public:
	BatchNormLayer() : Layer() {}
	
	void initialize(std::vector<size_t> input_shape) {};
	void forward() {};
	void backward(const Tensor& gradOutput) {}
};