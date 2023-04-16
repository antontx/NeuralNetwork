import numpy as np
import os

def vectorize_label( label):
        """
        Converts a numeric label (0-9) into a one-hot encoded vector representation.
        
        Args:
        label (int): the numeric label to be converted
        
        Returns:
        numpy.ndarray: a one-hot encoded vector representation of the label
        """
        e = np.zeros((10, 1))
        e[label] = 1
        return e
    
def convert(labels, images, size):
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
        label = vectorize_label(int.from_bytes(labelf.read(1),byteorder="big"))

        features = np.zeros((n,n))
        for row in range(n):
            for column in range(n):
                features[row][column] = int.from_bytes(imgf.read(1),byteorder="big")/255

        dataset.append([features.reshape(784,1),label])

    labelf.close()
    imgf.close()

    return dataset

def load( dataset_path):
    """
    Loads the MNIST training and test datasets from a directory path.
    
    Args:
    dataset_path (str): the path to the directory containing the MNIST dataset files
    
    Returns:
    tuple: a tuple containing two lists of feature-label pairs, representing the training and test datasets
    """
    training_images = os.path.join(dataset_path, "train-images.idx3-ubyte")
    training_labels = os.path.join(dataset_path, "train-labels.idx1-ubyte") 
    TRAINING_SIZE = 60000

    test_images = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")
    TEST_SIZE = 10000

    print("converting...")
    test_data = convert(test_labels, test_images, TEST_SIZE)
    training_data = convert(training_labels, training_images, TRAINING_SIZE)

    return training_data, test_data