# C++ Convolutional Neural Network (Still WIP)
This C++ project implements the workings of a Convolutional Neural Network from first principles (Entirely in header files for ease of implementation).
The network is a sequential system where, for each layer, it's input is a pointer to the output of a previous layer. 

This project was designed to be extremely explandable with abstract classes for `Optimizer`, `Loss` functions, and `Layer`. As a backbone for this project, 
A `Tensor` class was designed and implemented to store multi-dimensional data efficiently in `std::vector<float>` computing the stride of each dimension from 
the predifined shape of the `Tensor`. 
This project implements an SGD optimizer, ADAM optimizer (WIP), Cross Entropy loss function, and various layers described below. 

## Network API

The Network class brings everything together. When using it, there are a few steps to follow. An example usage could be: 
```cpp
Network network;

network.add(new ConvLayer(16, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
network.add(new PoolLayer(2, 2));
network.add(new ConvLayer(32, 3, 3, 1, 0, ActivationFunctions::TYPES::RELU));
network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
network.add(new PoolLayer(2, 2));
network.add(new FlattenLayer());
network.add(new DenseLayer(128, ActivationFunctions::TYPES::RELU));
network.add(new ActivationLayer(ActivationFunctions::TYPES::RELU));
network.add(new DenseLayer(10, ActivationFunctions::TYPES::SOFTMAX));
network.add(new ActivationLayer(ActivationFunctions::TYPES::SOFTMAX_CEL));

network.setInputShape({ BATCH_SIZE, 1, 28, 28 });
network.compile(new CrossEntropyLoss(), new SGD());
network.fit(test_data, test_labels, EPOCHS, BATCH_SIZE);
```

## Layer Classes

The various layers (listed below) are implemented following an abstract Layer class. This class requires the following functions to be implemented: 
```cpp
virtual void initialize(std::vector<size_t> input_shape) = 0;
virtual void forward() = 0;
virtual void backward(const Tensor& gradOutput) = 0;
```
Along with these functions, the abstract Layer class provides various relevent getters and setters. 
Specifiying the activation layer is necessary if an activation layer is directly after the current layer. 
The activation function specifies how the weights and biases shouold be implemented. 

### ConvLayer

Implements a convolutional layer with the following paramenters:
```cpp
ConvLayer(size_t num_filters, size_t filter_width, 
			  size_t filter_height, size_t stride = 1, size_t padding = 0, 
			  ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE)
```


### DenseLayer

Implements a dense layer with the following paramenters:
```cpp
DenseLayer(size_t output_size, ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE)
```

### PoolLayer

Implements a max pooling layer with the following paramenters:
```cpp
PoolLayer(size_t window_size, size_t stride = 1, ActivationFunctions::TYPES _ac = ActivationFunctions::TYPES::NONE)
```

### ActivationLayer

Implements an activation layer with the following paramenters:
```cpp
ActivationLayer(ActivationFunctions::TYPES _activation_function) : Layer(_activation_function) {}
```

### FlattenLayer

A simple layer to flatten the input from a tensor of shape `{batch size, D_1, ..., D_n}` to `{batch size, D_1 * ... * D_n}`.  
