import numpy as np
import DenseNeuralNetwork as dnn
from MNISTDataLoader import load
np.random.seed(1)


training_data, test_data = load("MNIST_dataset")

mlp = dnn.DenseNeuralNetwork([784, 32, 16, 10])

print("training")
mlp.train(training_data=training_data, epochs=20, batch_size=32, learning_rate=0.1, test_data=test_data, gd="mini-batch")


print("done!")
