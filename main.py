import numpy as np
import matplotlib.pyplot as plt
import DenseNeuralNetwork as dnn
from MNISTDataLoader import load
np.random.seed(1)


training_data, test_data = load("MNIST_dataset")

mlp = dnn.DenseNeuralNetwork([784, 32, 16, 10])

print("training")
mlp.train_mini_batch(training_data,epochs=30, batch_size=10, learning_rate=1, test_data=test_data)


print("done!")
