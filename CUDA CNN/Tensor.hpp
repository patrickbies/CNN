#pragma once

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <cmath>
#include <omp.h>

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

	inline float& operator()(size_t b, size_t c, size_t h, size_t w) {
		return data[b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
	}

	inline const float& operator()(size_t b, size_t c, size_t h, size_t w) const {
		return data[b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
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

#pragma omp parallel for
		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] * other.data[i];
		}

		return result;
	}

	Tensor operator/(const Tensor& other) const {
		if (shape != other.getShape()) {
			throw std::invalid_argument("Shape mismatch: Tensors must have the same shape for element-wise multiplication");
		}

		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] / other.data[i];
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

	Tensor operator/(float other) const {
		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] / other;
		}

		return result;
	}

	Tensor operator+(float other) const {
		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] + other;
		}

		return result;
	}

	Tensor operator+(const Tensor& other) const {
		Tensor result(shape);

		for (size_t i = 0; i < data.size(); i++) {
			result.data[i] = data[i] + other.data[i];
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

	void operator/=(float other) {
		for (size_t i = 0; i < data.size(); i++) {
			data[i] /= other;
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

	const std::vector<size_t>& getStrides() const {
		return strides;
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

	void zero() {
		for (size_t i = 0; i < data.size(); i++) data[i] = 0;
	}

	Tensor square() const {
		Tensor result(shape);
		for (size_t i = 0; i < data.size(); ++i) {
			result.data[i] = data[i] * data[i];
		}
		return result;
	}

	Tensor sqrt() const {
		Tensor result(shape);
		for (size_t i = 0; i < data.size(); ++i) {
			result.data[i] = std::sqrt(data[i]);
		}
		return result;
	}

	Tensor clamp(float a, float b) const {
		Tensor result(shape);
		for (size_t i = 0; i < data.size(); ++i) {
			result.data[i] = std::max(a, std::min(b, data[i]));
		}
		return result;
	}
};