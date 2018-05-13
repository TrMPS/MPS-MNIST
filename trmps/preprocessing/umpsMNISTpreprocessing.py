import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import sys
import os
from preprocessing.umpspreprocessing import *
#from rmpspreprocessing import *
from utils import spinner

from tensorflow.examples.tutorials.mnist import input_data

def _preprocess_images(data, size, selected_digit, shrink = True):
    """
    This function preprocesses images into format from the paper
    Supervised learning with quantum-inspired tensor networks

    :param data: tensorflow dataset
        The tensorflow dataset we are reading from
    :param size: integer
        The size of the dataset we wish to extract
    :param shrink: boolean
        Whether the image is shrunk using max pooling or not.
        If true, then the image is shrunk to 14x14 before being flattened.
        If false, the image is not shrunk.
    :return: (numpy array, numpy array)
        Returns (data points, results) in the format
        ([batch, MPS input size, other dimensions], [batch, classifications])
    """


    # written this way because originally, this was the only function and would read directly.
    # TODO: change all references to "mnist" with data
    mnist = data

    sess = tf.Session()
    data = []
    labels = []

    # Tensorflow operators / placeholders to resize the data from MNIST to format from paper
    # Resize images from 28*28 to 14*14
    image = tf.placeholder(tf.float32, shape=[784])
    if shrink:
        reshaped_image = tf.reshape(image, [-1, 28, 28, 1])
        pool = tf.nn.avg_pool(reshaped_image, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
        #pooled_image = tf.placeholder(tf.float32, shape=[1, 14, 14, 1])
        snaked_image = tf.reshape(pool, shape=[196])

        ones = tf.ones([196], dtype=tf.float32)
    else:
        snaked_image = image
        ones = tf.ones([784], dtype=tf.float32)
    phi = tf.stack([ones, snaked_image], axis=1)

    _spinner = spinner(jump = 300)

    # Loop through all the elements in the dataset and resize
    with sess.as_default():
        # sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter("output", sess.graph)
        # writer.close()
        counter = 0

        for i in range(20):
            percentage = int((i / 20) * 100)
            batch = mnist.next_batch(int(size / 20), shuffle=False)
            images = batch[0]
            for index, element in enumerate(images):
                #pooled = sess.run(pool,
                #                  feed_dict={reshaped_image: sess.run(reshaped_image,
                #                                                      feed_dict={image: element})})
                if batch[1][index][selected_digit] == 1 or batch[1][index][selected_digit+1] == 1:
                    data.append(np.array(sess.run(phi, feed_dict={image: element})))
                    labels.append(2 * (np.array(batch[1][index][selected_digit]) - 0.5))
                _spinner.print_spinner(percentage)
    _spinner.print_spinner(100.0)
    return (np.array(data), np.array(labels))

class UMPSMNISTDatasource(UMPSDatasource):
    """
    MNISTDatasource is a subclass of MPSDatasource which implements data loading for the
    well known MNIST dataset.

    Use as you would use any subclass of MPSDatasource.
    This class can also permute the MNIST images pixel-by-pixel.
    This class requires the use of tensorflow to load data.
    """

    def __init__(self, label_index, shrink=True, permuted=False, shuffled=False):
        """
        Initialises the dataset, and can also permute/shuffle the dataset.
        :param shrink: boolean
            Pass true to shrink the image to 14x14.
            If false, image will be kept at 28x28.
        :param permuted: boolean
            Pass true to have the image pixel-by-pixel permuted.
        :param shuffled: boolean
            Pass true to shuffle the dataset.
        """
        self.expected_shape = (784, 2)
        if shrink:
            self.expected_shape = (196, 2)
        self.shrink = shrink
        self.label_index = label_index
        expected_d_input = self.expected_shape[1]
        super().__init__(expected_d_input, shuffled)
        if permuted:
            print("permuting")
            data, labels = self._training_data[self.expected_shape[0]]
            print(len(data[0]))
            permutation = np.random.permutation(len(data[0]))
            permuted_data = []
            for d in data:
                permuted_data.append(np.array(d[permutation]))
            self._training_data = {self.expected_shape[0]: (np.array(permuted_data), labels)}
            test_data, test_labels = self._test_data[self.expected_shape[0]]
            permuted_test_data = []
            for d in test_data:
                permuted_test_data.append(np.array(d[permutation]))
            self._test_data = {self.expected_shape[0]: (np.array(permuted_test_data), test_labels)}

    def _load_test_data(self):
        """
        Loads test data of the appropriate size.
        :return: nothing
        """
        self._test_data = {self.expected_shape[0]: _preprocess_images(input_data.read_data_sets('MNIST_data', one_hot=True).test, size=10000, selected_digit=self.label_index, shrink=self.shrink)}
        super()._load_test_data()

    def _load_training_data(self):
        """
        Loads training data of the appropriate size.
        :return: nothing
        """
        self._training_data = {self.expected_shape[0]: _preprocess_images(input_data.read_data_sets('MNIST_data', one_hot=True).train,
                                                     size=60000, selected_digit=self.label_index, shrink=self.shrink)}
        super()._load_training_data()


if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    print("Testing")
    data_source = RMPSMNISTDatasource(shrink = False)
    (length, (data, labels)) = data_source.next_training_data_batch(1000, 784)
    print(length)
    print(data.shape)
    print(len(labels))