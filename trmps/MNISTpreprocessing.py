import tensorflow as tf
import numpy as np
import sys
import os
from preprocessing import *

from tensorflow.examples.tutorials.mnist import input_data

def _preprocess_images(data, size, shrink = True):
    # Function to process images into format from paper, also currently just returns the images for zeroes and ones,
    # so that we can create a binary classifier first.
    # Returned as a list of [images, classifications]


    # written this way because originally, this was the only function and would read directly.
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

    print("0 % done", end="")

    # Loop through all the elements in the dataset and resize
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()
        counter = 0

        for i in range(20):
            sys.stdout.flush()
            print("\r" + str(int((i / 20) * 100)) + " % done", end="")
            batch = mnist.next_batch(int(size / 20), shuffle=False)
            images = batch[0]
            for index, element in enumerate(images):
                #pooled = sess.run(pool,
                #                  feed_dict={reshaped_image: sess.run(reshaped_image,
                #                                                      feed_dict={image: element})})
                data.append(np.array(sess.run(phi, feed_dict={image: element})))
                labels.append(np.array(batch[1][index]))
                if index % 300 == 0:
                    # Spinner to show progress 
                    if counter == 0:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="|")
                        counter += 1
                    elif counter == 1:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="/")
                        counter += 1
                    elif counter == 2:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="-")
                        counter += 1
                    elif counter == 3:
                        print("\r" + str(int((i / 20) * 100)) + " % done", end="\\")
                        counter = 0
    sys.stdout.flush()
    print("\r" + str(100) + " % done")
    return (np.array(data), np.array(labels))
    
class MNISTDatasource(MPSDatasource):
    def __init__(self, shrink = True, permuted = False, shuffled = False):
        expected_shape = (784, 2)
        if shrink:
            expected_shape = (196,2)
        self.shrink = shrink
        super().__init__(expected_shape, shuffled)
        if permuted:
            print("permuting")
            data, labels = self._training_data
            print(len(data[0]))
            permutation = np.random.permutation(len(data[0]))
            permuted_data = []
            for d in data:
                permuted_data.append(np.array(d[permutation]))
            self._training_data = np.array(permuted_data), labels
            test_data, test_labels = self._test_data
            permuted_test_data = []
            for d in test_data:
                permuted_test_data.append(np.array(d[permutation]))
            self._test_data = np.array(permuted_test_data), test_labels
            
    def _load_test_data(self):
        self._test_data = _preprocess_images(input_data.read_data_sets('MNIST_data', one_hot=True).test, size=10000, shrink=self.shrink)
        super()._load_test_data()
    
    def _load_training_data(self):
        self._training_data = _preprocess_images(input_data.read_data_sets('MNIST_data', one_hot=True).train,
                                                     size=60000, shrink=self.shrink)
        super()._load_training_data()

if __name__ == "__main__":
    # If main, processes the images and also prints the number of images
    data_source = MNISTDatasource(shrink = False)
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(1000)
    print(data.shape)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(1000)
    print(len(labels))
    data, labels = data_source.next_training_data_batch(500)
