#include "Optimizer.hpp"
#include <unordered_map>
#include <iostream>

class Adam : public Optimizer {
private:
	float beta1;
	float beta2;
	float epsilon;
	size_t t;

	std::unordered_map<Tensor*, std::pair<Tensor, Tensor>> moments;	

public:
	Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-4)
		: Optimizer(lr), beta1(beta1), beta2(beta2),
		  epsilon(epsilon), t(0) {};

	void updateWeights(Tensor& weights, const Tensor& gradients) override {		
		if (!moments.count(&weights)) {
			initialize_moments(weights);
			std::cout << "initialized moments" << std::endl;
		}

		auto& a = moments[&weights];
		auto& m = a.first;
		auto& v = a.second;

		t++;

		m = m * beta1 + gradients * (1 - beta1);
		v = v * beta2 + gradients.square() * (1 - beta2);

		float bias_correction1 = 1 - std::pow(beta1, t);
		float bias_correction2 = 1 - std::pow(beta2, t);

		Tensor m_hat = m * (1.0f / bias_correction1);
		Tensor v_hat = v * (1.0f / bias_correction2);

		// Apply update
		weights -= (m_hat * learning_rate) / (v_hat.sqrt() + epsilon);
	};

private: 
	void initialize_moments(Tensor& param) {
		moments[&param] = std::make_pair(
			Tensor(param.getShape()),
			Tensor(param.getShape())
		);
	}
};