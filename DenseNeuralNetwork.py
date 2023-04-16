import matplotlib.pyplot as plt
import numpy as np
import random


class DenseNeuralNetwork():
    def __init__(self, layer_sizes) -> None:
        self.layer_sizes = layer_sizes
        self.layer_count = len(layer_sizes)  # = L

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

        # dimensions(activations)

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
                    f"Epoch {epoch+1}: {self.evaluate(test_data)}")

    def train_sgd(self, training_data, learning_rate=None, epochs=None, test_data=None):
        for x, y in training_data:  # for every example in the data set
                Δb, ΔW = self.backprop(x, y)  # calculate the gradient
                # update the parameters
                self.update_parameters(ΔW, Δb, learning_rate)

    def train_batch(self, training_data, learning_rate=None, epochs=None, test_data=None):
        nabla_b = np.array([np.zeros(b.shape)
                            for b in self.biases], dtype=object)
        nabla_w = np.array([np.zeros(w.shape)
                            for w in self.weights], dtype=object)

        for x, y in training_data:  # for every example in the data set
            Δb, ΔW = self.backprop(x, y)  # calculate the gradient

            nabla_b += Δb
            nabla_w += ΔW

        nabla_b /= len(training_data)
        nabla_w /= len(training_data)

        # update the parameters
        self.update_parameters(nabla_w, nabla_b, learning_rate)

    def train_mini_batch(self, training_data, batch_size=32, learning_rate=0.1, epochs=250, test_data=None):
        nabla_b = np.array([np.zeros(b.shape)
                            for b in self.biases], dtype=object)
        nabla_w = np.array([np.zeros(w.shape)
                            for w in self.weights], dtype=object)

        # shuffle the training data for each epoch
        random.shuffle(training_data)

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            batch_size = len(batch)

            for x, y in batch:
                # calculate the gradient for the current example
                delta_b, delta_w = self.backprop(x, y)
                nabla_b += delta_b
                nabla_w += delta_w

            # calculate the average gradient over the mini-batch
            nabla_b /= batch_size
            nabla_w /= batch_size

            self.update_parameters(nabla_w, nabla_b, learning_rate)

    def backprop(self, x, y):
        L = self.layer_count - 2
        ΔW = np.array([np.zeros(w.shape) for w in self.weights], dtype=object)
        Δb = np.array([np.zeros(b.shape) for b in self.biases], dtype=object)
        δ = [0] * (L+1)

        _, a, z = self.feed_forward(x, True)

        δ[L] = np.multiply(a[-1]-y, sigmoid(z[-1], True))

        Δb[L] = δ[L]
        ΔW[L] = np.dot(δ[L], a[L].transpose())

        for l in range(L-1, -1, -1):

            δ[l] = np.multiply(
                np.dot(self.weights[l+1].transpose(), δ[l+1]), sigmoid(z[l], True))
            Δb[l] = δ[l]
            ΔW[l] = np.dot(δ[l], a[l].transpose())

        return Δb, ΔW
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            if np.argmax(y) == np.argmax(self.feed_forward(x)):
                correct += 1
        return f"{correct}/{len(test_data)}"

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))
    for arr in arrays_list:
        print(arr.shape)
    print()
