import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import time
from mps.mps_sgd import SGDMPS
import utils
from tensorflow.python.client import timeline
import pickle

class SGDOptimizer(object):
    """
    SGDOptimizer is used to optimise SGDMPS using stochastic gradient descent.
    The stochastic gradient descent is implemented with the tensorflow AdamOptimizer.

    Variables:
    MPS: SGDMPS
        An SGDMPS object to be optimised

    Usage example:

    # Parameters
    d_feature = 2
    d_output = 10
    batch_size = 100
    permuted = False
    shuffled = True
    shrink = True
    input_size = 196

    rate_of_change = 0.001
    feature_reg=1.1
    reg=0.1/batch_size
    n_step = 1200

    data_source = MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

    # Make the SGDMPS network and initialise with linear regression
    network = SGDMPS(d_feature, d_output, input_size,
                        feature_reg=feature_reg,
                        reg=reg)
    network.prepare(data_source)

    # Optimise with stochastic gradient descent
    optimizer = SGDOptimizer(network)
    optimizer.train(data_source, batch_size, n_step,
                    rate_of_change=rate_of_change)


    """

    def __init__(self, network):
        """
        Initialise the optimizer
        :param MPSNetwork: SGDMPS
            THE MPS network to be optimised
        """
        self.MPS = network
        print( "_____   Thomas the Tensor Train    . . . . . o o o o o",
           "  __|[_]|__ ___________ _______    ____      o",
           " |[] [] []| [] [] [] [] [_____(__  ][]]_n_n__][.",
           "_|________|_[_________]_[________]_|__|________)<",
           "  oo    oo 'oo      oo ' oo    oo 'oo 0000---oo\_",
           " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", sep="\n")

    def train(self, data_source, batch_size, n_steps, rate_of_change=0.001):
        """
        Trains the network.
        The weights are saved under weights_sgd as a pickle file at the end of the training

        :param data_source: (some subclass of) MPSDatasource
            The data/labels that the MPS will be trained on
        :param batch_size: integer
            The batch size used for one step of stochastic gradient descent
        :param n_steps: integer
            Number of batches to pass to the MPS
        :param rate_of_change: float
            Rate of change to pass into the AdamOptimizer

        """
        features = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        labels = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])

        f = self.MPS.predict(features)
        cost = self.MPS.cost(f, labels)
        accuracy = self.MPS.accuracy(f, labels)
        conf_mat = self.MPS.confusion_matrix(f, labels)
        trainer = tf.train.AdamOptimizer(rate_of_change).minimize(cost)

        to_eval = [accuracy, cost, conf_mat]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            start = time.time()
            for i in range(n_steps):

                batch_feature, batch_label = data_source.next_training_data_batch(batch_size)
                feed_dict = {features: batch_feature, labels: batch_label}
                _ = sess.run(trainer, feed_dict=feed_dict)

                if i % 100 == 0:
                    end = time.time()
                    train_acc, train_cost, train_conf_mat = sess.run(to_eval, feed_dict=feed_dict)
                    print('step {}, accuracy {}'.format(i, train_acc))
                    print('step {}, training cost {}'.format(i, train_cost))
                    print(train_conf_mat)
                    print('each step takes {}s on average'.format((end-start)/100))
                    start = time.time()

            test_feature, test_label = data_source.test_data
            feed_dict = {features: test_feature, labels: test_label}
            to_eval = [self.MPS.nodes, accuracy, cost, conf_mat]
            weights, test_acc, test_cost, test_conf_mat = sess.run(to_eval, feed_dict=feed_dict)
            print('testing accuracy {}'.format(test_acc))
            print('testing cost {}'.format(test_cost))
            print(test_conf_mat)

            with open('weights_sgd', 'wb') as fp:
                pickle.dump(weights, fp)

# if __name__ == '__main__':
#     # Model parameters
#     d_feature = 2
#     d_output = 10
#     batch_size = 100
#     permuted = False
#     shuffled = True
#     shrink = True
#     input_size = 784
#     if shrink:
#         input_size = 196

#     rate_of_change = 0.001
#     feature_reg=1.1
#     reg=0.1/batch_size
#     n_step = 1200

#     data_source = MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

#     weights = None

#     network = SGDMPS(d_feature, d_output, input_size,
#                         feature_reg=feature_reg,
#                         reg=reg)
#     network.prepare(data_source)
#     optimizer = SGDOptimizer(network)
#     optimizer.train(data_source, batch_size, n_step,
#                     rate_of_change=rate_of_change)



