#include "Tensor.hpp"

class Optimizer {
protected:
    float learning_rate;

public:
    Optimizer(float lr = 0.001) : learning_rate(lr) {}

    virtual ~Optimizer() = default;

    virtual void updateWeights(Tensor& weights, const Tensor& gradients) = 0;

    virtual void updateBiases(Tensor& biases, const Tensor& gradients) {
        updateWeights(biases, gradients);
    }

    virtual void setLearningRate(float lr) {
        learning_rate = lr;
    }

    virtual float getLearningRate() const {
        return learning_rate;
    }
};