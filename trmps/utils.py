import tensorflow as tf
import urllib
import tarfile
import sys

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
            print("\r" + str(100) + " % done")
        elif self.index % self.jump == 0:
            sys.stdout.flush()
            # Spinner to show progress 
            if self.counter == 0:
                print("\r" + str(percentage) + " % done", end="|")
                self.counter += 1
            elif self.counter == 1:
                print("\r" + str(percentage) + " % done", end="/")
                self.counter += 1
            elif self.counter == 2:
                print("\r" + str(percentage) + " % done", end="-")
                self.counter += 1
            elif self.counter == 3:
                print("\r" + str(percentage) + " % done", end="\\")
                self.counter = 0
        self.index += 1







