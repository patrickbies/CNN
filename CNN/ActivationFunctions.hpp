#include "Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

class ActivationFunctions {
private: 
    static float sig(float a) {
        return 1.0f / (1.0f + exp(-a));
    }

public:
    enum TYPES {
        RELU,
        SOFTMAX,
        SIGMOID,
        NONE
    };

    // ReLU activation
    static Tensor relu(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) { 
            return std::max(0.0f, b); 
        });
        return result;
    }

    // Derivative of ReLU
    static Tensor relu_derivative(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) { 
            return b > 0.0f ? 1.0f : 0.0f; 
        });
        return result;
    }

    // Sigmoid activation
    static Tensor sigmoid(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) {return sig(b); });
        return result;
    }

    // Derivative of Sigmoid:
    static Tensor sigmoid_derivative(const Tensor& a) {
        Tensor result = a;
        result.apply([](float b) { 
            return sig(b) * (1 - sig(b));
        });
        return result;
    }

    // Softmax activation
    static Tensor softmax(const Tensor& a) {
        Tensor result = a;
        std::vector<size_t> shape = a.getShape();
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        float max_value = *std::max_element(a.data.begin(), a.data.end());

        float summation = 0.0f;
        for (size_t i = 0; i < size; i++) {
            result.data[i] = std::exp(a.data[i] - max_value);
            summation += result.data[i];
        }

        for (size_t i = 0; i < size; i++) {
            result.data[i] /= summation;
        }

        return result;
    }
    
    // Derivative of softmax (diagonals of jacobian): 
    static Tensor softmax_derivative(const Tensor& a) {
        Tensor _sm = softmax(a);
        Tensor res = Tensor(a.getShape());

        for (size_t i = 0; i < a.data.size(); i++) {
            res.data[i] = _sm.data[i] * (1 - _sm.data[i]);
        }

        return res;
    }

    // Jacobian of softmax
    static Tensor softmax_jacobian(const Tensor& a) {
        Tensor softmax_output = softmax(a);
        std::vector<size_t> shape = a.getShape();
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

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
