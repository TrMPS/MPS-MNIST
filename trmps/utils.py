import tensorflow as tf
import urllib
import tarfile
import sys
import numpy as np
import matplotlib.pyplot as plt

def check_nan(tensor, name, replace_nan=True):
    s = tf.reduce_sum(tensor)
    is_nan = tf.is_nan(s)
    tensor = tf.cond(is_nan,
                     true_fn=lambda: tf.Print(tensor, [tensor],
                                            message='{} is not finite'.format(name)),
                     false_fn=lambda: tensor)
    if replace_nan:
        tensor = tf.where(tf.is_nan(tensor),
                          tf.zeros_like(tensor),
                          tensor)
    return tensor

def getunzipped(url, name):
    try:
        name, hdrs = urllib.request.urlretrieve(url, name)
    except IOError as e:
        print('Cannot retrieve {}: {}'.format(url, e))
        return
    z = tarfile.open(name, "r:gz")
    z.extractall()
    z.close()
    print('Data downloaded and unzipped')

def list_from(tensorArray, length):
    """
    list_from is a helper function that produces a list from a tensorArray.
    It is used to extract the results of training in MPSOptimizer.
    :param tensorArray: tensorflow TensorArray
        The tensor array that is to be converted to a list
    :param length: integer
        The length of the TensorArray/the list that is to be created
    :return: list of tensorflow Tensors
        A list containing all the values of the TensorArray as Tensors.
        This has to then be evaluated to get actual values.
    """
    arr = tensorArray
    result_list = []
    with tf.name_scope("createlist"):
        for i in range(length):
            result_list.append(arr.read(i))
    return result_list


class spinner(object):
    def __init__(self, jump = 400):
        self.index = 0
        self.jump = jump
        self.percentage = 0
        self.counter = 0

    def print_spinner(self, percentage):
        if float(percentage) == 100.0:
            sys.stdout.flush()
            print("\r" + str(100) + " % done      ")
        elif self.index % self.jump == 0:
            percentage_str = "{:.2f}".format(percentage)
            sys.stdout.flush()
            # Spinner to show progress
            if self.counter == 0:
                print("\r" + percentage_str + " % done", end="|")
                self.counter += 1
            elif self.counter == 1:
                print("\r" + percentage_str + " % done", end="/")
                self.counter += 1
            elif self.counter == 2:
                print("\r" + percentage_str + " % done", end="-")
                self.counter += 1
            elif self.counter == 3:
                print("\r" + percentage_str + " % done", end="\\")
                self.counter = 0
        self.index += 1

# Adapted from https://stackoverflow.com/questions/29831489/numpy-1-hot-array
def convert_to_onehot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def into_image(snaked_image):
    length = len(snaked_image)
    dim = int(np.sqrt(length))
    if float(dim) != float(np.sqrt(length)):
        print("something's wrong; please pass in a snaked image that was originally square")
    image = np.reshape(snaked_image, [dim, dim])
    return image

def show(snaked_image):
    imgplot = plt.imshow(into_image(snaked_image), interpolation='none', cmap='binary')


def plot_func(optimizer, costs1, costs2, i):
    start = optimizer.MPS._special_node_loc
    to_plot = costs1[start:]
    to_plot = np.append(to_plot, costs2[::-1])
    to_plot = np.append(to_plot, costs1[:start])
    print(start, optimizer.MPS.input_size)
    plt.plot(to_plot)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()



