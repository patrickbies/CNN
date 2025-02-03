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
        for (int i = 0; i < a.data.size(); i++) {
            location.data[i] = std::max(0.0f, a.data[i]);
        }
    }

    // Derivative of ReLU
    static void relu_derivative(Tensor& location, const Tensor& a) {
        for (int i = 0; i < a.data.size(); i++) {
            location.data[i] = a.data[i] > 0.0f ? 1.0f : 0.0f;
        }
    }

    // Sigmoid activation
    static void sigmoid(Tensor& location, const Tensor& a) {
        for (int i = 0; i < a.data.size(); i++) {
            location.data[i] = sig(a.data[i]);
        }
    }

    // Derivative of Sigmoid:
    static void sigmoid_derivative(Tensor& location, const Tensor& a) {
        for (int i = 0; i < a.data.size(); i++) {
            location.data[i] = sig(a.data[i]) * (1 - sig(a.data[i]));
        }
    }

    // Softmax activation
    static void softmax(Tensor& location, const Tensor& a) {
        std::vector<size_t> shape = a.getShape();
        size_t batch_size = shape[0];
        size_t num_logits = shape[1];

        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float max_val = *std::max_element(a.data.begin() + batch_idx * num_logits,
                a.data.begin() + (batch_idx + 1) * num_logits);
            max_val = std::min(std::max(max_val, -50.0f), 50.0f); 

            float sum = 0.0f;

            for (size_t i = 0; i < num_logits; i++) {
                size_t idx = batch_idx * num_logits + i;
                float val = std::exp(a.data[idx] - max_val);
                val = std::min(std::max(val, 1e-20f), 1e20f); 
                location.data[idx] = val;
                sum += val;
            }

            sum = std::max(sum, 1e-20f);
            for (size_t i = 0; i < num_logits; i++) {
                size_t idx = batch_idx * num_logits + i;
                location.data[idx] /= sum;
            }
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
