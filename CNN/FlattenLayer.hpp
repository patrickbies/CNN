#pragma once

#include "Layer.hpp"
#include <stdexcept>

class FlattenLayer : public Layer {
private:
    std::vector<size_t> input_shape; 

public:
    FlattenLayer() : Layer(ActivationFunctions::TYPES::NONE) {}

    void initialize(std::vector<size_t> input_shape) override {
        if (input_shape.size() < 2) {
            throw std::invalid_argument("Input shape must have at least two dimensions.");
        }

        this->input_shape = input_shape;

        size_t flattened_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>()) / input_shape[0];

        input_gradient = new Tensor(input_shape);
        output = new Tensor({ input_shape[0], flattened_size });
    }

    void forward() override {
        if (!input) {
            throw std::runtime_error("Input tensor is not set for FlattenLayer.");
        }

        output->data = input->data;
    }

    void backward(const Tensor& gradOutput) override {
        if (gradOutput.data.size() != input->data.size()) {
            throw std::invalid_argument("Gradient output size must match input size for FlattenLayer.");
        }

        input_gradient->data = gradOutput.data; 
    }
};
