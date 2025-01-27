#include "MNISTToTensor.hpp"
#include "Network.hpp"
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "PoolLayer.hpp"
#include "FlattenLayer.hpp"
#include "CrossEntropyLoss.hpp"
#include "SGD.hpp"

const char* input_file = "mnist_train.csv";
const char* test_file = "mnist_test.csv";
const size_t BATCH_SIZE = 32;
const size_t NUM_BATCHES = 10;
const size_t EPOCHS = 10;

// DELETE LATER : Stole from Network class
void setBatch(Tensor& batch_tensor, const Tensor& data, size_t batch, size_t batch_size) {
	const size_t total_elements_per_batch = batch_tensor.data.size();
	const size_t start_idx = batch * batch_size * total_elements_per_batch;
	const size_t end_idx = start_idx + total_elements_per_batch;

	if (end_idx > data.data.size()) {
		throw std::out_of_range("Batch index exceeds data size.");
	}

	std::copy(data.data.begin() + start_idx,
		data.data.begin() + end_idx,
		batch_tensor.data.begin());
}

int main() {
	// data pair contains: { data, labels }
	std::pair<Tensor, Tensor> data = MNISTToTensor::parseCSV(input_file);
	// std::pair<Tensor, Tensor> test_data = MNISTToTensor::parseCSV(test_file);

	Tensor test_data = Tensor({ BATCH_SIZE * NUM_BATCHES, 1, 28, 28 });
	Tensor test_labels = Tensor({ BATCH_SIZE * NUM_BATCHES, 10 });

	setBatch(test_data, data.first, 0, BATCH_SIZE * NUM_BATCHES);
	setBatch(test_labels, data.second, 0, BATCH_SIZE * NUM_BATCHES);

	Network network = Network();

	network.add(new ConvLayer(16, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new PoolLayer(2, 2));
	network.add(new ConvLayer(32, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new PoolLayer(2, 2));
	network.add(new FlattenLayer());
	network.add(new DenseLayer(128, ActivationFunctions::TYPES::RELU));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
	network.add(new DenseLayer(10, ActivationFunctions::TYPES::SOFTMAX));
	network.add(new ActivationLayer(ActivationFunctions::TYPES::SOFTMAX_CEL));

	//network.setInputShape({ 1, 1, 28, 28 });
	//network.compile(new CrossEntropyLoss(), new SGD());
	//std::cout << "test before training: " << network.test(test_data.first, test_data.second) << std::endl;

	network.setInputShape({ BATCH_SIZE, 1, 28, 28 });
	network.compile(new CrossEntropyLoss(), new SGD());
	network.fit(test_data, test_labels, EPOCHS, BATCH_SIZE);

	//network.setInputShape({ 1, 1, 28, 28 });
	//network.compile(new CrossEntropyLoss(), new SGD());
	//std::cout << "test after training: " << network.test(test_data.first, test_data.second) << std::endl;

	return 0;
}