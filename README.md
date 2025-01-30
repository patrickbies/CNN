# C++ Convolutional Neural Network
> neural network system built in C++

This project implements the workings of a Convolutional Neural Network from first principles.

This project was designed to be explandable with abstract classes for `Optimizer`, `Loss`, and `Layer`. As a backbone for this project, 
A `Tensor` class was designed and implemented to store multi-dimensional data efficiently in `std::vector<float>` computing the stride of each dimension from 
the predifined shape of the `Tensor`. 

As currently implemented, this project contains an SGD optimizer, ADAM optimizer, Cross Entropy loss function, and various [layers](#layer-classes) described below. 
The main.cpp in this repository, contains a demo set up to train a network on the MNIST dataset (Including an MNISTToTensor.hpp which parses the MNIST data). 

### TODO

- [x] Rewrite codebase to accept varying batch size without reinitializing weights and biases
- [ ] Write an export function in Network class
- [ ] Move Tensor implementation to its own project, optimize tensor operations
- [ ] Speed up convolutions and pooling by implementing im2col algorithm
- [ ] Rewrite layers using CUDA for GPU based training
- [ ] Graphical interface to visualize training process

## Network API

The Network is a sequential system where, for each layer, it's input is a pointer to the output of a previous layer. 
This class is used to create a network, train it, test it, and eventually export it. 

An example usage could be: 
```c
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

### Member functions: 

`void add(Layer* layer)`
Adds a layer to the network.

`void setInputShape(std::vector<size_t> _input_shape)`
Sets the shape of the input data.

`void compile(Loss* _loss_function, Optimizer* _optimizer)`
Compiles the network by setting the loss function and optimizer and initializing all layers based on the input shape.
- Throws an exception if the input shape is not set.

`void linkLayers(size_t batches)`
Initializes the layers and sets up input/output relationships for a given batch size.
- Throws an exception if the input shape is not set.

`Tensor* step(size_t ind)`
Performs a forward pass through a single layer.
- Throws an exception if layers are not added, input is not set, or network is not compiled.

`void fit(const Tensor& training_data, const Tensor& labels, size_t epochs, size_t batch_size)`
Trains the network using the given training data and labels over a specified number of epochs and batch size.

`float one_hot_accuracy(const Tensor& training_data, const Tensor& labels)`
Computes the accuracy of the network using one-hot encoding for classification.

`Tensor* predict(Tensor* input)`
Runs the forward pass through the entire network and returns the final output.

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
