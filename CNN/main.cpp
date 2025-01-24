#include "MNISTToTensor.hpp"
#include "Network.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "PoolLayer.hpp"
#include "CrossEntropyLoss.hpp"
#include "SGD.hpp"

const char* input_file = "mnist_train.csv";

int main() {
	std::pair<Tensor, Tensor> data = MNISTToTensor::parseCSV(input_file);

	Network network = Network();

	network.add(new ConvLayer(8, 1, 3, 3, 1, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new PoolLayer(2));
	network.add(new DenseLayer(128, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new DenseLayer(10, ActivationFunctions::TYPES::SOFTMAX));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::SOFTMAX_CEL));

	network.compile(new CrossEntropyLoss(), new SGD());
	network.fit(data.first, data.second, 16, 8);
}