import numpy as np
import os

def one_hot(label,vector_size):
        """
        Converts a numeric label (0-9) into a one-hot encoded vector representation.
        
        Args:
        label (int): the numeric label to be converted
        
        Returns:
        numpy.ndarray: a one-hot encoded vector representation of the label
        """
        vector = np.zeros((vector_size, 1))
        vector[label] = 1
        return vector
    
def convert(labels, images, size, output_vector_size):
    """
    Converts MNIST dataset files into a list of feature-label pairs.
    
    Args:
    labels (str): the path to the MNIST labels file
    images (str): the path to the MNIST images file
    size (int): the number of examples to convert
    
    Returns:
    list: a list of feature-label pairs, where each feature is a numpy array of image pixels and each label is a one-hot encoded vector representation
    """
    labelf = open(labels, "rb") #rb -> reading the file as binary
    imgf = open(images, "rb")

    #start locations of the actual data
    labelf.read(8) 
    imgf.read(16)

    n = 28

    dataset = []
    for i in range(size):
        label = one_hot(int.from_bytes(labelf.read(1),byteorder="big"),output_vector_size)

        features = np.zeros((n,n))
        for row in range(n):
            for column in range(n):
                features[row][column] = int.from_bytes(imgf.read(1),byteorder="big")/255

        dataset.append([features.reshape(784,1),label])

    labelf.close()
    imgf.close()

    return dataset

def load(training_images_path, training_labels_path, test_images_path, test_labels_path, training_size, test_size, output_vector_size):
    """
    Loads the MNIST training and test datasets from the given file paths.

    Args:
    training_images_path (str): the path to the training images file
    training_labels_path (str): the path to the training labels file
    test_images_path (str): the path to the test images file
    test_labels_path (str): the path to the test labels file
    training_size (int): the size of the training dataset
    test_size (int): the size of the test dataset

    Returns:
    tuple: a tuple containing two lists of feature-label pairs, representing the training and test datasets
    """

    test_data = convert(test_labels_path, test_images_path, test_size, output_vector_size)
    training_data = convert(training_labels_path, training_images_path, training_size, output_vector_size)

    return training_data, test_data
