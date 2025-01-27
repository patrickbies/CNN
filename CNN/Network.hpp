#pragma once

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include <iostream>

class Network {
private: 
	std::vector<Layer*> layers;
	Loss* loss_function = nullptr;
	Optimizer* optimizer = nullptr;

	std::vector<size_t> input_shape;
	Tensor batch_input;
	Tensor batch_labels;

public: 
	void add(Layer* layer) {
		layers.push_back(layer);
	}

	void setInputShape(std::vector<size_t> _input_shape) {
		input_shape = _input_shape;
	}

	void compile(Loss* _loss_function, Optimizer* _optimizer) {
		if (!input_shape.size())
			throw std::exception("input shape must be set.");
		loss_function = _loss_function;
		optimizer = _optimizer;

		std::vector<size_t> next_shape = input_shape;
		Tensor* next_input = nullptr;

		for (size_t i = 0; i < layers.size(); i++) {
			layers[i]->initialize(next_shape);
			layers[i]->setInput(next_input);

			next_input = layers[i]->getOutput();
			next_shape = layers[i]->getOutput()->getShape();
		}
	}

	// Returns a pointer tensor incase debugging is needed; 
	Tensor* step(size_t ind) {
		if (ind >= layers.size() || !layers[ind]->getInput())
			throw std::exception("Must add layers, or must set input, or must compile network.");

		layers[ind]->forward();
		return layers[ind]->getOutput();
	}

	void fit(const Tensor& training_data, const Tensor& labels, size_t epochs, size_t batch_size) {
		std::vector<size_t> bi_shape = training_data.getShape(), bl_shape = labels.getShape();
		bi_shape[0] = bl_shape[0] = batch_size;

		batch_input = Tensor(bi_shape);
		batch_labels = Tensor(bl_shape);

		for (size_t i = 0; i < epochs; i++) {
			train_epoch(training_data, labels, batch_size);
			std::cout << "Epoch " << i + 1 << " completed." << std::endl;
		}
	}
	
	Tensor* predict(Tensor* input) {
		layers[0]->setInput(input);

		for (size_t i = 0; i < layers.size(); i++) {
			step(i);
		}

		return layers.back()->getOutput();
	}

private:
	// debug function:
	void printVec(std::vector<float> s) {
		for (auto& a : s) std::cout << a << std::endl;
	}

	void printVec(std::vector<size_t> s) {
		for (auto& a : s) std::cout << a << std::endl;
	}

	void backward(Tensor& loss_gradient) {
		Tensor* current = &loss_gradient;
		for (size_t i = layers.size(); i-- > 0;) {
			layers[i]->backward(*current);

			if (layers[i]->getWeightGradient() != nullptr) {
				optimizer->updateWeights(layers[i]->weights, *layers[i]->getWeightGradient());
			}
			if (layers[i]->getBiasGradient() != nullptr) {
				optimizer->updateBiases(layers[i]->biases, *layers[i]->getBiasGradient());
			}

			current = layers[i]->getInputGradient();
		}
	}
	
	void train_epoch(const Tensor& data, const Tensor& labels, size_t batch_size) {
		size_t num_batches = data.getShape()[0] / batch_size;

		for (size_t i = 0; i < num_batches; i++) {
			setBatch(batch_input, data, i, batch_size);

			Tensor* predictions = predict(&batch_input);
			setBatch(batch_labels, labels, i, batch_size);

			Tensor loss_gradient = loss_function->backward(batch_labels, *predictions);
			std::cout << "Error from batch " << i << ": " << loss_function->compute(batch_labels, *predictions) << std::endl;
			backward(loss_gradient);
		}
	}

	void setBatch(Tensor& batch_tensor, const Tensor& data, size_t batch, size_t batch_size) {
		const size_t total_elements_per_batch = batch_tensor.data.size();
		const size_t start_idx = batch * total_elements_per_batch;
		const size_t end_idx = start_idx + total_elements_per_batch;

		if (end_idx > data.data.size()) {
			throw std::out_of_range("Batch index exceeds data size.");
		}

		std::copy(data.data.begin() + start_idx,
			data.data.begin() + end_idx,
			batch_tensor.data.begin());
	}
};