import numpy as np
import sys
import os
from preprocessing.MNISTpreprocessing import MNISTDatasource
from utils import convert_to_onehot, spinner
import gzip
from enum import Enum
import skimage.measure

# TODO: Make it so that FashionMNISTDatasource downloads the data automatically.

class data_type(Enum):
    training = 0
    testing = 1

def load_mnist(path, data_type=data_type.training):
    """
    Copied from
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py,
    But with edits so that it no longer "snakes" the image
    """

    print("Extracting...")
    kind = "train"
    if data_type == data_type.testing:
        kind = "t10k"
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    if os.path.isfile(labels_path):
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 28, 28)
    else:
        exception_str = ("Please place the MNIST Data in directory\n" + str(path) +
                         "\nWith the names\n" + ('%s-images-idx3-ubyte.gz' % kind) +
                         "\nAnd\n" + ('%s-labels-idx1-ubyte.gz' % kind))
        raise Exception(exception_str)
    # print("label shape: ", labels.shape)
    # print("sample label:", labels[0])
    return images, labels

class FashionMNISTDatasource(MNISTDatasource):
    """
    FashionMNISTDatasource is a drop-in replacement for MNISTDatasource with the Fashion MNIST
    Dataset instead of the MNIST dataset.
    http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    """
    def _load_with_correct_shape(self, data_type):
        raw_images, raw_labels = load_mnist(type(self).__name__, data_type)
        raw_images = raw_images / 255.0
        if self.shrink == True:
            new_images = []
            for image in raw_images:
                new_images.append(skimage.measure.block_reduce(image, (2, 2), np.max))
            raw_images = np.array(new_images)
        current_shape = raw_images.shape
        chain_length = current_shape[1] * current_shape[2]
        ones = np.ones(chain_length)
        raw_data = raw_images.reshape(current_shape[0], chain_length)
        _spinner = spinner(jump=300)
        data = []
        for element in raw_data:
            new_element = np.column_stack((ones, element))
            data.append(new_element)
        data = np.array(data)
        print(data.shape)
        labels = convert_to_onehot(raw_labels)
        return (data, labels)

    def _load_test_data(self):
        """
        Loads test data of the appropriate size.
        :return: nothing
        """
        self._test_data = self._load_with_correct_shape(data_type.testing)
        super()._save_test_data()

    def _load_training_data(self):
        """
        Loads training data of the appropriate size.
        :return: nothing
        """
        self._training_data = self._load_with_correct_shape(data_type.training)
        super()._save_training_data()

if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    data_source = FashionMNISTDatasource(shrink=True)
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(data[:,0,1])
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(1000)
    print(len(labels))
    print(labels[0])
    data, labels = data_source.next_training_data_batch(500)
