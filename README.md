# NeuralNetworks

This is an implementation of a dense neural network class in Python. It is mainly based on the book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com) by Michael Nielsen.

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
```

---

## `train(training_data, gd='mini-batch', batch_size=32, learning_rate=0.1, epochs=250, test_data=None, interval=None)`

Trains the neural network on the given `training_data`. 

#### Parameters:
- `training_data`: list of tuples `(x, y)` representing the training inputs and corresponding desired outputs.
- `gd`: string indicating the gradient descent method to use. Possible values are `'batch'`, `'mini-batch'`, and `'sgd'`. Default is `'mini-batch'`.
- `batch_size`: integer representing the number of training examples in each mini-batch. Only used if `gd` is set to `'mini-batch'`. Default is `32`.
- `learning_rate`: float representing the learning rate used for updating the weights and biases during training. Default is `0.1`.
- `epochs`: integer representing the number of epochs (iterations over the entire training data set) to run during training. Default is `250`.
- `test_data`: list of tuples `(x, y)` representing the testing inputs and corresponding desired outputs. If provided, the network's accuracy on the testing data will be evaluated after each epoch.
- `interval`: integer representing the number of epochs between each evaluation of the network's accuracy on the testing data. Default is `epochs // 100` if `epochs >= 100`, otherwise `5`.

#### Returns:
None.

---

## `feed_forward(a, cache=False)`

Runs a feedforward pass through the neural network with the given input activations `a`.

#### Parameters:
- `a`: 1D numpy array representing the input activations for the first layer of the neural network.
- `cache`: boolean indicating whether to return intermediate values useful for backpropagation. Default is `False`.

#### Returns:
- If `cache=False`, returns a 1D numpy array representing the output activations of the neural network.
    - to devectorize the output of the neural network, one can pass the output activations array through the argmax() function which is included in the Numpy library
    ```python
    output = np.argmax(network.feed_forward(x))
    ```

- If `cache=True`, returns a tuple `(output_activations, activations, weighted_inputs)`, where:
    - `output_activations`: 1D numpy array representing the output activations of the neural network.
    - `activations`: list of 1D numpy arrays representing the activations of all layers of the neural network, including the input layer.
    - `weighted_inputs`: list of 1D numpy arrays representing the weighted inputs to all layers of the neural network, excluding the input layer.

## `load(filename)`

Loads the parameters of a trained neural network from a CSV file.

#### Parameters:
- `filename`: string representing the filename (including path if necessary) of the CSV file to load the parameters from.

#### Returns:
A `DenseNeuralNetwork` object with the weights and biases loaded from the CSV file.

---

## `save(filename)`

Saves the parameters of a trained neural network to a CSV file.

#### Parameters:
- `filename`: string representing the filename (including path if necessary) of the CSV file to save the parameters to.

#### Returns:
None. The parameters are saved to the specified CSV file.

