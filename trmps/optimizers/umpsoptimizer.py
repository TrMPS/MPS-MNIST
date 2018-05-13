import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import time
import numpy as np
# from mps.umps import UMPS
# from preprocessing.rmpspreprocessing import UMPSDatasource
import pickle
import utils
from tensorflow.python.client import timeline
from optimizers.parameterObjects import MPSOptimizerParameters, MPSTrainingParameters

def dummy_length(current_iter):
    return 10

class UMPSOptimizer(object):
    def __init__(self, UMPS, max_size, default_length=None, optional_parameters=MPSOptimizerParameters()):
        self.parameters = optional_parameters
        self.path = self.parameters.path
        self.umps = UMPS
        self.rate_of_change = tf.placeholder(tf.float32, shape=[])
        self._length = tf.placeholder(tf.float32, shape=[])
        self._batch_size = tf.placeholder(tf.float32, shape=[])
        """
        TODO: Currently unsupported features from the optimiser for normal MPS:
        - Using the Hessian
        - Armijo
        - Multiple updates per step
        - Support a logistic loss as well as a quadratic one
        - min_singular_value
        """
        self.verbosity = self.parameters.verbosity
        self.min_singular_value = 0.0
        self.max_size = max_size
        self.current_length = default_length
        self._feature = tf.placeholder(tf.float32, shape=[None, None, self.umps.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None])
        self._train_step()
        print("_____   Thomas the Tensor Train Mk.2   . . . o o o o o",
              "  __|[_]|__ ___________ _______    ____      o",
              " |[] [] []| [] [] [] [] [_____(__  ][]]_n_n__][.",
              "_|________|_[_________]_[________]_|__|________)<",
              "  oo    oo 'oo      oo ' oo    oo 'oo 0000---oo\_",
              " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", sep="\n")

    def _train_step(self):
        batch_size = tf.shape(self._feature)[1]
        length = tf.shape(self._feature)[0]
        self.length = length
        with tf.name_scope('calculate_C1'):

            C1 = tf.tensordot(self._feature[0], self.umps.weight_left, [[1], [0]])
            cond = lambda c, *args: tf.less(c, self.umps.centre_loc)
            _, C1, _ = tf.while_loop(cond=cond, body=self._chain_multiply_r, loop_vars=[1, C1, self._feature],
                                        shape_invariants=[tf.TensorShape(
                                            []), tf.TensorShape([None, None, None]),
                                        tf.TensorShape(None)],
                                        parallel_iterations=5)

        with tf.name_scope('calculate_C2'):
            C2 = tf.tensordot(self._feature[-1], self.umps.weight_right, [[1], [0]])
            cond2 = lambda c, *args: tf.greater(c, self.umps.centre_loc)
            _, C2, _ = tf.while_loop(cond=cond2, body=self._chain_multiply_l, loop_vars=[self.length - 1, C2, self._feature],
                                  shape_invariants=[tf.TensorShape(
                                      []), tf.TensorShape([None, None, None]),
                                  tf.TensorShape(None)],
                                  parallel_iterations=5)
        contracted_sp_node = tf.einsum(
            'nij,tn->tij', self.umps.weight_centre, self._feature[self.umps.centre_loc])
        C1_sp = tf.einsum('tij,tjk->tik', C1, contracted_sp_node)
        all_contracted = tf.einsum('tij,tjk->tik', C1_sp, C2)
        predictions = tf.einsum('tij,tij->t', all_contracted, all_contracted)
        diff = predictions - self._label
        contracted_l_node = tf.einsum(
            'nij,tn->tij', self.umps.weight_left, self._feature[self.umps.centre_loc])
        C1_c = tf.einsum('tij,tjk->tik', C1, contracted_l_node)
        dfdwc = tf.einsum('til,tkj,tn,tij->tnlk', C1, C2, self._feature[self.umps.centre_loc], all_contracted)
        dfdc = tf.einsum('til,tkj,tij->tlk', C1_c, C2, all_contracted)
        wc_gradient = -1 * tf.einsum('t,tnlk->nlk', diff, dfdwc) # / tf.cast(batch_size, tf.float32)
        c_gradient = -1 * tf.einsum('t,tlk->lk', diff, dfdc) # / tf.cast(batch_size, tf.float32)
        self._update_w_c(wc_gradient)
        self._update_c(c_gradient)
        self._update_w_l()
        self._update_w_r()

    def _update_w_c(self, wc_gradient):
        wc_gradient = tf.Print(wc_gradient, [wc_gradient])
        self._updated_w_c = self.umps.weight_centre + self.rate_of_change * wc_gradient

    def _update_c(self, c_gradient):
        self._updated_c = self.umps._c + self.rate_of_change * c_gradient

    def _update_w_l(self):
        _left = tf.einsum('mij,kj->mik', self._updated_w_c, self._updated_c)
        _left_dims = tf.shape(_left)
        _left_flattened = tf.reshape(_left, [_left_dims[0] * _left_dims[1], _left_dims[2]])
        s, u, v = tf.svd(_left_flattened)
        filtered_u = u
        filtered_v = v
        uv = tf.matmul(filtered_u, filtered_v)
        _w_l = tf.reshape(uv, [_left_dims[0], _left_dims[1], _left_dims[2]])
        self._updated_w_l = _w_l

    def _update_w_r(self):
        _right = tf.einsum('ik,mij->mkj', self._updated_c, self._updated_w_c)
        _right_dims = tf.shape(_right)
        _right_flattened = tf.reshape(_right, [_right_dims[0] * _right_dims[1], _right_dims[2]])
        s, u, v = tf.svd(_right_flattened)
        filtered_u = u
        filtered_v = v
        uv = tf.matmul(filtered_u, filtered_v)
        _w_r = tf.reshape(uv, [_right_dims[0], _right_dims[1], _right_dims[2]])
        self._updated_w_r = _w_r

    def _chain_multiply_l(self, counter, C2, feature):
        with tf.name_scope('chain_multiply_left'):
            node = self.umps.weight_right
            contracted_node = tf.tensordot(feature[counter], node,
                                           [[1], [0]])
            contracted_node.set_shape([None, None, None])
            C2.set_shape([None, None, None])
            C2 = tf.einsum('tij,tjl->til', contracted_node, C2)
            counter = counter - 1

        return [counter, C2, feature]

    def _chain_multiply_r(self, counter, C1, feature):
        with tf.name_scope('chain_multiply_right'):
            node = self.umps.weight_left
            contracted_node = tf.tensordot(feature[counter], node,
                                           [[1], [0]])
            contracted_node.set_shape([None, None, None])
            C1.set_shape([None, None, None])
            C1 = tf.einsum('tli,tij->tlj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1, feature]

    def _test_step(self, feature, label):
        """
        A single step of training
        :param self:
        :param feature:
        :param label:
        :return:
        """
        f = self.umps.predict(feature)
        cost = self.umps.cost(f, label)
        accuracy = self.umps.accuracy(f, label)
        # confusion = self.umps.confusion_matrix(f, label)
        return cost, accuracy, f #, confusion

    def train(self, data_source, batch_size, n_step, length=None, optional_parameters=MPSTrainingParameters()):
        """
        length should be some function that returns the length of the sequence to be trained on based on the current iteration
        """
        #TODO: Doesn't support learning rate changing in training
        if length is None and self.current_length is None:
            raise ValueError("Need either default_length to be an integer or length to be a function that returns an integer")
        elif self.current_length is None:
            self.current_length = length(0)
        run_options = []
        run_metadata = []
        if optional_parameters._logging_enabled:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        self.feed_dict = None
        optimised_w_c = None
        optimised_c = None
        optimised_w_l = None
        optimised_w_r = None
        lr = optional_parameters.rate_of_change
        train_cost, train_accuracy, _ = self._test_step(self._feature, self._label)
        actual_length, (test_feature, test_label) = data_source.test_data_of_length(self.current_length)
        print("Length of test data:", actual_length)

        feature = tf.placeholder(tf.float32, shape=[None, None, self.umps.d_feature])
        label = tf.placeholder(tf.float32, shape=[None])
        test_cost, test_accuracy, f = self._test_step(feature, label)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_step):
                start = time.time()
                if length is not None:
                    self.current_length = length(i)
                actual_length, (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size, self.current_length)
                print("Length of training data:", actual_length)
                self.feed_dict = self.umps.create_feed_dict(optimised_w_c, optimised_c, optimised_w_l, optimised_w_r)
                self.feed_dict[self._feature] = batch_feature
                self.feed_dict[self._label] = batch_label
                self.feed_dict[self.rate_of_change] = lr
                self.feed_dict[feature] = test_feature
                self.feed_dict[label] = test_label
                self.feed_dict[self._length] = np.float32(actual_length)
                self.feed_dict[self._batch_size] = np.float32(batch_size)
                to_eval = [train_cost, train_accuracy, test_cost, test_accuracy,
                           self._updated_w_c, self._updated_c, self._updated_w_l, self._updated_w_r, f]
                train_c, train_acc, test_c, test_acc, optimised_w_c, optimised_c, optimised_w_l, optimised_w_r, prediction = sess.run(to_eval,
                                                                                                                  feed_dict=self.feed_dict,
                                                                                                                  options=run_options,
                                                                                                                  run_metadata=run_metadata)
                end = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S"))
                print('step {}, training cost {}, accuracy {}. Took {} s'.format(i, train_c, train_acc, end - start))
                print('step {}, testing cost {}, accuracy {}'.format(i, test_c, test_acc))
                # print('step {}, optimised_c {}, optimised_w_c{}, optimised_w_l{}, optimised_w_r{}, predictions{}'.format(i, optimised_c, optimised_w_c, optimised_w_l, optimised_w_r, prediction))
                print('step {}, predictions{}, shapes{}, {}'.format(i, prediction, optimised_w_c.shape, optimised_c.shape))
                # print('confusion matrix: \n' + str(test_conf))





