#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include <iostream>

class Network {
private: 
	std::vector<Layer*> layers;
	Loss* loss_function = nullptr;
	Optimizer* optimizer = nullptr;

public: 
	void add(Layer* layer) {
		layers.push_back(layer);
	}

	void compile(Loss* _loss_function, Optimizer* _optimizer) {
		loss_function = _loss_function;
		optimizer = _optimizer;

		Tensor* prev_out = nullptr;

		for (size_t ind = 0; ind < layers.size(); ind++) {
			layers[ind]->setInput(prev_out);
			prev_out = layers[ind]->getOutput();
		}
	}

	void setInput(Tensor* _input) {
		if (layers.size()) layers[0]->setInput(_input);
		else throw std::exception("Must add layers.");
	}

	// Returns a pointer tensor incase debugging is needed; 
	Tensor* step(size_t ind) {
		if (ind >= layers.size() || !layers[ind]->getInput())
			throw std::exception("Must add layers, or must set input, or must compile network.");

		layers[ind]->forward();
		return layers[ind]->getInput();
	}

	void fit(const Tensor& training_data, const Tensor& labels, size_t epochs, size_t batch_size) {
		for (size_t i = 0; i < epochs; i++) {
			train_epoch(training_data, labels, batch_size);
			std::cout << "Epoch " << i + 1 << " completed." << std::endl;
		}
	}

	Tensor* predict(const Tensor& input) {
		for (size_t i = 0; i < layers.size(); i++) {
			step(i);
		}

		return layers.back()->getOutput();
	}

private:
	void train_epoch(const Tensor& data, const Tensor& labels, size_t batch_size) {
		size_t num_batches = data.getShape()[0] / batch_size;

		for (size_t i = 0; i < num_batches; i++) {
			Tensor* predictions = predict(getBatch(data, i, batch_size));

			float loss_gradient = loss_function->compute(labels, *predictions);
		}
	}

	Tensor getBatch(const Tensor& data, size_t batch, size_t batch_size) {
		return data.linear_slice(batch * batch_size, batch * (batch_size + 1));
	}
};