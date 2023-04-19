import matplotlib.pyplot as plt
import numpy as np
import random

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))

class DenseNeuralNetwork():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)



        self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        

    def feed_forward(self, a, cache=False):
        activations = [a]
        weighted_inputs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = sigmoid(z)

            activations.append(a)
            weighted_inputs.append(z)

        if not cache:
            return a
        return a, activations, weighted_inputs

    def update_parameters(self, ΔW, Δb, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * ΔW[i]
            self.biases[i] -= learning_rate * Δb[i]

    def train(self, training_data, gd = "mini-batch",batch_size= 32, learning_rate=0.1, epochs=250, test_data=None, interval = None):
        if interval == None:
            interval = epochs // 100 if epochs >= 100 else 5

        for epoch in range(epochs):
            if gd == "mini-batch":
                self.train_mini_batch(training_data, batch_size, learning_rate, epochs, test_data)
            elif gd == "batch":
                self.train_batch(training_data, learning_rate, epochs, test_data)
            elif gd == "sgd":
                self.train_sgd(training_data, learning_rate, epochs, test_data)
            else:
                raise Exception("Invalid Gradient Descent Method")

            if epoch % interval == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}: {self.evaluate(test_data)} MSE: {np.mean(self.MeanSquaredError(test_data))}")
                
    def train_plot(self, training_data, gd = "mini-batch",batch_size= 32, learning_rate=0.1, epochs=250, test_data=None, interval = None):
        if interval == None:
            interval = epochs // 100 if epochs >= 100 else 5

        out = [[np.mean(self.MeanSquaredError(test_data))],[self.evaluate(test_data)]]

        for epoch in range(epochs):
            if gd == "mini-batch":
                self.train_mini_batch(training_data, batch_size, learning_rate, epochs, test_data)
            elif gd == "batch":
                self.train_batch(training_data, learning_rate, epochs, test_data)
            elif gd == "sgd":
                self.train_sgd(training_data, learning_rate, epochs, test_data)
            else:
                raise Exception("Invalid Gradient Descent Method")

            if epoch % interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} MSE: {np.mean(self.MeanSquaredError(test_data))}")
                out[0].append(np.mean(self.MeanSquaredError(test_data)))
                out[1].append(self.evaluate(test_data))

    def train_sgd(self, training_data, learning_rate=None, epochs=None, test_data=None):
        for x, y in training_data:  # for every example in the data set
                bias_gradients, weight_gradients = self.backprop(x, y)  # calculate the gradient
                # update the parameters
                self.update_parameters(weight_gradients, bias_gradients, learning_rate)

    def train_batch(self, training_data, learning_rate=None, epochs=None, test_data=None):
        bias_gradients = np.array([np.zeros(b.shape)
                            for b in self.biases], dtype=object)
        weight_gradients = np.array([np.zeros(w.shape)
                            for w in self.weights], dtype=object)

        for x, y in training_data:  # for every example in the data set
            bias_gradients_single, weight_gradients_single = self.backprop(x, y)  # calculate the gradient

            bias_gradients += bias_gradients_single
            weight_gradients += weight_gradients_single

        bias_gradients /= len(training_data)
        weight_gradients /= len(training_data)

        # update the parameters
        self.update_parameters(weight_gradients, bias_gradients, learning_rate)

    def train_mini_batch(self, training_data, batch_size=32, learning_rate=0.1, epochs=250, test_data=None):
        bias_gradients = np.array([np.zeros(b.shape)
                            for b in self.biases], dtype=object)
        weight_gradients = np.array([np.zeros(w.shape)
                            for w in self.weights], dtype=object)

        # shuffle the training data for each epoch
        random.shuffle(training_data)

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            batch_size = len(batch)

            for x, y in batch:
                # calculate the gradient for the current example
                bias_gradients_single, weight_gradients_single = self.backprop(x, y)
                bias_gradients += bias_gradients_single
                weight_gradients += weight_gradients_single

            # calculate the average gradient over the mini-batch
            bias_gradients /= batch_size
            weight_gradients /= batch_size

            self.update_parameters(weight_gradients, bias_gradients, learning_rate)

    def backprop(self, x, y):
        last_layer = self.layer_count - 2
        weights_gradients = np.array([np.zeros(w.shape) for w in self.weights], dtype=object)
        bias_gradients = np.array([np.zeros(b.shape) for b in self.biases], dtype=object)
        E = [0] * (last_layer+1)

        _, a, z = self.feed_forward(x, True)

        E[last_layer] = np.multiply(a[-1]-y, sigmoid(z[-1], True))

        bias_gradients[last_layer] = E[last_layer]
        weights_gradients[last_layer] = np.dot(E[last_layer], a[last_layer].transpose())

        for layer in range(last_layer-1, -1, -1):

            E[layer] = np.multiply(
                np.dot(self.weights[layer+1].transpose(), E[layer+1]), sigmoid(z[layer], True))
            bias_gradients[layer] = E[layer]
            weights_gradients[layer] = np.dot(E[layer], a[layer].transpose())

        return bias_gradients, weights_gradients

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            if np.argmax(y) == np.argmax(self.feed_forward(x)):
                correct += 1
        return f"{correct}/{len(test_data)}"

    def MeanSquaredError(self, test_data):
        mse = 0
        for x, y in test_data:
            y_hat = self.feed_forward(x)
            mse += (y_hat-y)**2
        return mse/len(test_data)
