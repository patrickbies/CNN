#include "Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

class ActivationFunctions {
public:
    // ReLU activation
    static Tensor relu(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) { return std::max(0.0f, b); });
        return result;
    }

    // Derivative of ReLU
    static Tensor relu_derivative(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) { return b > 0.0f ? 1.0f : 0.0f; });
        return result;
    }

    // Softmax activation
    static Tensor softmax(const Tensor& a) {
        Tensor result = a;
        std::vector<size_t> shape = a.getShape();
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        // Find the maximum value for numerical stability
        float max_value = *std::max_element(a.data.begin(), a.data.end());

        // Compute exponentials and their sum
        float summation = 0.0f;
        for (size_t i = 0; i < size; i++) {
            result.data[i] = std::exp(a.data[i] - max_value);
            summation += result.data[i];
        }

        // Normalize
        for (size_t i = 0; i < size; i++) {
            result.data[i] /= summation;
        }

        return result;
    }

    // Jacobian of softmax
    static Tensor softmax_derivative(const Tensor& a) {
        Tensor softmax_output = softmax(a);
        std::vector<size_t> shape = a.getShape();
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        // create a jacobian matrix
        Tensor jacobian({ size, size });

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                if (i == j) {
                    jacobian({ i, j }) = softmax_output.data[i] * (1 - softmax_output.data[i]);
                }
                else {
                    jacobian({ i, j }) = -softmax_output.data[i] * softmax_output.data[j];
                }
            }
        }

        return jacobian;
    }
};
