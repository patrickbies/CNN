#include "Loss.hpp"
#include <cmath>

class CrossEntropyLoss : public Loss {
public: 
	float compute(const Tensor& labels, const Tensor& predictions) override {
		float loss = 0.0f;

		if (labels.getShape() != predictions.getShape()) {
			throw std::out_of_range("Labels and Predictions size do not match.");
		}
		
		for (size_t i = 0; i < labels.getShape()[0]; i++) {
			for (size_t j = 0; j < labels.getShape()[1]; j++) {
				loss += labels({ i, j }) * std::log(std::max(1e-8f, predictions({ i, j })));
			}
		}

		return -loss / labels.getShape()[0];
	};

	Tensor backward(const Tensor& labels, const Tensor& predictions) override {
		if (labels.getShape() != predictions.getShape()) {
			throw std::out_of_range("Labels and Predictions size do not match.");
		}

		return predictions - labels;
	};
};