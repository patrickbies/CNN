#include "Tensor.hpp"

class Loss {
    virtual ~Loss() {}

    virtual float compute(const Tensor& labels, const Tensor& predictions) = 0;

    virtual Tensor backward(const Tensor& labels, const Tensor& predictions) = 0;
};