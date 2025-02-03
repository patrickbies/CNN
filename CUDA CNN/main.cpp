#include "MNISTToTensor.hpp"
#include "Network.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "PoolLayer.hpp"
#include "FlattenLayer.hpp"
#include "BatchNormLayer.hpp"
#include "DropoutLayer.hpp"
#include "CrossEntropyLoss.hpp"
#include "SGD.hpp"
#include "Adam.hpp"

const char* input_file = "mnist_train.csv";
const char* test_file = "mnist_test.csv";
const size_t BATCH_SIZE = 60;
const size_t EPOCHS = 10;

// DELETE LATER : Stole from Network class
void setBatch(Tensor& batch_tensor, const Tensor& data, size_t batch, size_t batch_size) {
	size_t start_idx = batch * batch_size * data.getStrides()[0];
	size_t end_idx = (batch + 1) * batch_size * data.getStrides()[0];

	if (end_idx > data.data.size()) {
		throw std::out_of_range("batch out of range");
	}

	std::copy(data.data.begin() + start_idx,
		data.data.begin() + end_idx,
		batch_tensor.data.begin());
}

int main() {
	// data pair contains: { data, labels }
	std::pair<Tensor, Tensor> data = MNISTToTensor::parseCSV(input_file);
	std::pair<Tensor, Tensor> test = MNISTToTensor::parseCSV(test_file);

	size_t NUM_BATCHES = data.first.getShape()[0] / BATCH_SIZE;

	Tensor test_data = Tensor({ BATCH_SIZE * NUM_BATCHES, 1, 28, 28 });
	Tensor test_labels = Tensor({ BATCH_SIZE * NUM_BATCHES, 10 });

	setBatch(test_data, data.first, 0, BATCH_SIZE * NUM_BATCHES);
	setBatch(test_labels, data.second, 0, BATCH_SIZE * NUM_BATCHES);

	Network network;

	network.add(new ConvLayer(32, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	network.add(new ConvLayer(32, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	network.add(new PoolLayer(2, 2));
	//network.add(new DropoutLayer(0.25));

	network.add(new ConvLayer(64, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	network.add(new ConvLayer(64, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	network.add(new PoolLayer(2, 2));
	//network.add(new DropoutLayer(0.25));

	network.add(new FlattenLayer());
	network.add(new DenseLayer(512, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	//network.add(new DropoutLayer(0.25));

	network.add(new DenseLayer(1024, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	//network.add(new BatchNormLayer());
	//network.add(new DropoutLayer(0.5));

	network.add(new DenseLayer(10, ActivationFunctions::TYPES::SOFTMAX));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::SOFTMAX_CEL));

	network.setInputShape({ 1, 28, 28 }); // CWH no batch size included
	network.compile(new CrossEntropyLoss(), new Adam());

	network.fit(test_data, test_labels, EPOCHS, BATCH_SIZE, [&network, &test]() {
		std::cout << "current validation: " << network.one_hot_accuracy(test.first, test.second) << std::endl;
	});

	return 0;
}