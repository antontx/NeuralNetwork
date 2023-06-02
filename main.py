import numpy as np
import DenseNeuralNetwork as dnn
from MNISTDataLoader import load
np.random.seed(1)

training_data, test_data = load("MNIST_dataset")

mlp = dnn.DenseNeuralNetwork([784, 32, 16, 10])
mlp.train(training_data=training_data, epochs=250, batch_size=32, learning_rate=0.5, test_data=test_data, gd="mini-batch")
mlp.save("parameters2.csv")

untrained_mlp = dnn.DenseNeuralNetwork([784, 32, 16, 10])
print(untrained_mlp.evaluate(test_data))
untrained_mlp.load("parameters2.csv")
print(untrained_mlp.evaluate(test_data))



