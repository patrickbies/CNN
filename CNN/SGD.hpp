#pragma once

#include "Optimizer.hpp"

class SGD : public Optimizer {	
public: 
	SGD(float learning_rate = 0.0001) : Optimizer(learning_rate) {};

	void updateWeights(Tensor& weights, const Tensor& gradients) override {
		weights -= gradients * learning_rate;
	};
};