#pragma once
#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>

class Tensor {
private: 
	std::vector<float> data;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	void computeStrides();
	size_t flatten(const std::vector<size_t>& indices) const;

public: 
	Tensor() = default;

	// next few functions and operators implemented in header for ease:
	Tensor(const std::vector<size_t> shape, float initial = 0.0f) : shape(shape) {
		computeStrides();
		data.resize(strides[0] * shape[0], initial);
	}

	float& operator()(const std::vector<size_t>& indices) {
		return data[flatten(indices)];
	}

	const float& operator()(const std::vector<size_t>& indices) const {
		return data[flatten(indices)];
	}

	const std::vector<size_t>& getShape() const {
		return shape;
	}

	Tensor reshape(const std::vector<size_t> newShape);
	void apply(std::function<float(float)> func);
};