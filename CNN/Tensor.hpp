#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>

class Tensor {
private: 
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	void computeStrides() {
		strides.resize(shape.size());
		size_t stride = 1;

		for (int i = shape.size() - 1; i >= 0; i--) {
			strides[i] = stride;
			stride *= shape[i];
		}
	}

	size_t flatten(const std::vector<size_t>& indices) const {
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

public: 
	std::vector<float> data;

	Tensor() = default;

	Tensor(const std::vector<size_t> shape, float initial = 0.0f) : shape(shape) {
		computeStrides();
		data.resize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()), initial);
	}

	float& operator()(const std::vector<size_t>& indices) {
		return data[flatten(indices)];
	}

	const float& operator()(const std::vector<size_t>& indices) const {
		return data[flatten(indices)];
	}

	Tensor operator*(const Tensor& other) const {
		if (shape != other.getShape()) {
			throw std::invalid_argument("Shape mismatch: Tensors must have the same shape for element-wise multiplication");
		}

		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] * other.data[i];
		}

		return result;
	}

	Tensor operator*(float other) const {
		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] * other;
		}

		return result;
	}

	void operator-=(const Tensor& other) {
		if (shape != other.getShape()) {
			throw std::invalid_argument("Shape mismatch: Tensors must have the same shape for element-wise subtraction");
		}

		for (size_t i = 0; i < data.size(); i++) {
			data[i] -= other.data[i];
		}
	}

	const Tensor operator-(const Tensor& other) const {
		if (shape != other.getShape()) {
			throw std::invalid_argument("Shape mismatch: Tensors must have the same shape for element-wise subtraction");
		}

		Tensor result = Tensor(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] - other.data[i];
		}

		return result;
	}

	const std::vector<size_t>& getShape() const {
		return shape;
	}

	void reshape(const std::vector<size_t> newShape) {
		size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
		if (newSize != data.size()) {
			throw std::invalid_argument("New shape must have the same total size as the old shape");
		}

		shape = newShape;
		computeStrides();
	}

	Tensor& apply(std::function<float(float)> func) {
		for (auto& a : Tensor::data) a = func(a);
		return *this;
	}

	Tensor linear_slice(size_t start, size_t end) const {
		std::vector<size_t> new_shape = shape;
		new_shape[0] = end - start;

		Tensor result = Tensor(new_shape);

		size_t slice_size = std::accumulate(
			shape.begin() + 1, shape.end(), 1, std::multiplies<>());

		size_t result_idx = 0;
		for (size_t i = start; i < end; ++i) {
			size_t source_offset = i * slice_size;
			std::copy(
				data.begin() + source_offset,
				data.begin() + source_offset + slice_size,
				result.data.begin() + result_idx);
			result_idx += slice_size;
		}

		return result;
	}
};