#include "MNISTToTensor.hpp"
#include "Network.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "PoolLayer.hpp"
#include "CrossEntropyLoss.hpp"
#include "SGD.hpp"

const char* input_file = "mnist_train.csv";
const char* test_file = "mnist_test.csv";
const size_t BATCH_SIZE = 64;
const size_t EPOCHS = 50;

int main() {
	// data pair contains: { data, labels }
	std::pair<Tensor, Tensor> data = MNISTToTensor::parseCSV(input_file);
	// std::pair<Tensor, Tensor> test_data = MNISTToTensor::parseCSV(test_file);

	Network network = Network();

	network.add(new ConvLayer(32, 1, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new PoolLayer(2, 2));
	network.add(new DenseLayer(128, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new DenseLayer(10, ActivationFunctions::TYPES::SOFTMAX));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::SOFTMAX_CEL));

	//network.setInputShape({ 1, 1, 28, 28 });
	//network.compile(new CrossEntropyLoss(), new SGD());
	//std::cout << "test before training: " << network.test(test_data.first, test_data.second) << std::endl;

	network.setInputShape({ BATCH_SIZE, 1, 28, 28 });
	network.compile(new CrossEntropyLoss(), new SGD());
	network.fit(data.first, data.second, EPOCHS, BATCH_SIZE);

	//network.setInputShape({ 1, 1, 28, 28 });
	//network.compile(new CrossEntropyLoss(), new SGD());
	//std::cout << "test after training: " << network.test(test_data.first, test_data.second) << std::endl;

	return 0;
}