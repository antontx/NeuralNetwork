# NeuralNetworks

This is an implementation of a dense neural network class in Python. It is based on the book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com) by Michael Nielsen.

The `DenseNeuralNetwork` class can be used to create a neural network with an arbitrary number of layers and neurons per layer. It can be trained using three different gradient descent methods: mini-batch, batch, and stochastic gradient descent.

## Requirements

This code requires the following packages to be installed:

- `numpy`
- `random`

## Usage

The `DenseNeuralNetwork` class can be used as follows:

```python
# create a neural network with two input neurons, one hidden layer with four neurons, and one output neuron
network = DenseNeuralNetwork([2, 4, 1])

# train the neural network on some training data
network.train(training_data)

# evaluate the neural network on some test data
accuracy = network.evaluate(test_data)
