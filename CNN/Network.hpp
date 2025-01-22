#include "Layer.hpp"

class Network {
private: 
	std::vector<Layer*> layers;
public: 
	void add(Layer& layer) {
		layers.push_back(&layer);
	}

};