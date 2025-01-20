#include "ActivationFunctions.hpp"
#include <random>

class Initializer {
public: 
	// Uniform distribution (will be used as default unless activation funtion is specified):
	static float uniform(const size_t fan_in) {
		std::random_device rd;
		std::mt19937 gen(rd());
		float limit = std::sqrt(1.0f / static_cast<float>(fan_in));
		std::uniform_real_distribution<float> dist(-limit, limit);

		return dist(gen);
	}

	// He initialization for ReLU activation:
	static float he_init(const size_t fan_in) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> dist{ 0, std::sqrt(2.0f / static_cast<float>(fan_in)) };

		return dist(gen);
	}

	// Xavier initialization for Sigmoid or Softmax:
	static float xavier_init(const size_t fan_in, const size_t fan_out) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> dist{ 0, std::sqrt(2.0f / (static_cast<float>(fan_in) + static_cast<float>(fan_out))) };

		return dist(gen);
	}
};