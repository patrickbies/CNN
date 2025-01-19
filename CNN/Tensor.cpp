#include "Tensor.hpp"

void Tensor::computeStrides() {
	strides.resize(shape.size());
	size_t stride = 1;

	for (int i = shape.size() - 1; i >= 0; i--) {
		strides[i] = stride;
		stride *= shape[i];
	}
}

size_t Tensor::flatten(const std::vector<size_t>& indices) const {
	if (indices.size() != shape.size()) {
		throw std::invalid_argument("Number of indices must match tensor rank");
	}
	size_t idx = 0;
	for (int i = 0; i < shape.size(); i++) {
		if (indices[i] >= shape[i]) {
			throw std::out_of_range("Index out of range.");
		}
		idx += strides[i] * indices[i];
	}

	return idx;
}

Tensor Tensor::reshape(const std::vector<size_t> newShape)
{
	size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
	if (newSize != data.size()) {
		throw std::invalid_argument("New shape must have the same total size as the old shape");
	}

	Tensor::shape = newShape;
	Tensor::computeStrides();
}

void Tensor::apply(std::function<float(float)> func) 
{
	for (auto& a : Tensor::data) a = func(a);
}