import tensorflow as tf
import time
from mps import MPS
import pickle
import utils
from tensorflow.python.client import timeline
from optimizer import MPSOptimizer

class SDGOptimizer(object):

	def __init__(self):
		self.MPS = MPSNetwork
        self.rate_of_change = tf.placeholder(tf.float32, shape=[])
        self.max_size = max_size
        self.grad_func = grad_func
        self.cutoff = cutoff
        self._feature = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self._setup_optimization()
        _ = self.train_step()


    def _setup_optimization(self):
    	"""
    	load all the nodes from placeholder to a list
    	"""
    	

    def train_step(): 
