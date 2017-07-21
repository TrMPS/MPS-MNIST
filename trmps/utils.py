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







