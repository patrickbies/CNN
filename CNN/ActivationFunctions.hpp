#pragma once

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
        SOFTMAX_CEL, // should be used when a Cross Entropy Loss will be used, and this is the last layer
        NONE
    };

    // ReLU activation
    static void relu(Tensor& location, const Tensor& a) {
        location.apply([](float b) {
            return std::max(0.0f, b); 
        });
    }

    // Derivative of ReLU
    static void relu_derivative(Tensor& location, const Tensor& a) {
        location.apply([](float b) {
            return b > 0.0f ? 1.0f : 0.0f; 
        });
    }

    // Sigmoid activation
    static void sigmoid(Tensor& location, const Tensor& a) {
        location.apply([](float b) {return sig(b); });
    }

    // Derivative of Sigmoid:
    static void sigmoid_derivative(Tensor& location, const Tensor& a) {
        location.apply([](float b) { 
            return sig(b) * (1 - sig(b));
        });
    }

    // Softmax activation
    static void softmax(Tensor& location, const Tensor& a) {
        std::vector<size_t> shape = a.getShape();

        float max_value = *std::max_element(a.data.begin(), a.data.end());

        float summation = 0.0f;
        for (size_t i = 0; i < a.data.size(); i++) {
            location.data[i] = std::exp(a.data[i] - max_value);
            summation += location.data[i];
        }

        for (size_t i = 0; i < a.data.size(); i++) {
            location.data[i] /= summation;
        }
    }
    
    // Derivative of softmax (diagonals of jacobian): 
    static void softmax_derivative(Tensor& location, const Tensor& a) {
        Tensor _sm = Tensor(a.getShape());
        softmax(_sm, a);

        for (size_t i = 0; i < a.data.size(); i++) {
            location.data[i] = _sm.data[i] * (1 - _sm.data[i]);
        }
    }

    // Jacobian of softmax
    static Tensor softmax_jacobian(const Tensor& a) {
        Tensor _sm = Tensor(a.getShape());
        softmax(_sm, a);

        Tensor jacobian({ a.data.size(), a.data.size() });

        for (size_t i = 0; i < a.data.size(); i++) {
            for (size_t j = 0; j < a.data.size(); j++) {
                if (i == j) {
                    jacobian({ i, j }) = _sm.data[i] * (1 - _sm.data[i]);
                }
                else {
                    jacobian({ i, j }) = -_sm.data[i] * _sm.data[j];
                }
            }
        }

        return jacobian;
    }
};
