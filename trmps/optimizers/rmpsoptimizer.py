import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import time
import numpy as np
from mps.rmps import RMPS
from preprocessing.rmpspreprocessing import RMPSDatasource
import pickle
import utils
from tensorflow.python.client import timeline
from optimizers.parameterObjects import MPSOptimizerParameters, MPSTrainingParameters

def dummy_length(current_iter):
    return 10

class RMPSOptimizer(object):
    def __init__(self, RMPS, max_size, default_length=None, optional_parameters=MPSOptimizerParameters()):
        self.parameters = optional_parameters
        self.path = self.parameters.path
        self.RMPS = RMPS
        self.rate_of_change = tf.placeholder(tf.float32, shape=[])
        """
        TODO: Currently unsupported features from the optimiser for normal MPS:
        - Using the Hessian
        - Armijo
        - Multiple updates per step
        - Support a logistic loss as well as a quadratic one
        """
        self.verbosity = self.parameters.verbosity
        self.min_singular_value = self.parameters.min_singular_value
        self.max_size = max_size
        self.current_length = default_length
        self._feature = tf.placeholder(tf.float32, shape=[None, None, self.RMPS.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None, self.RMPS.d_output])
        self._setup_optimization(self._feature, self._label)
        self.train_step(self._feature, self._label)
        print("_____   Thomas the Tensor Train Mk.2   . . . o o o o o",
              "  __|[_]|__ ___________ _______    ____      o",
              " |[] [] []| [] [] [] [] [_____(__  ][]]_n_n__][.",
              "_|________|_[_________]_[________]_|__|________)<",
              "  oo    oo 'oo      oo ' oo    oo 'oo 0000---oo\_",
              " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", sep="\n")

    def _setup_optimization(self, feature, labels):
        batch_size = tf.shape(feature)[1]
        length = tf.shape(feature)[0]
        self.length = length
        cond = lambda c, *args: tf.less(c, length)
        condr = lambda c, *args: tf.greater(c, -1)
        with tf.name_scope('setup_optimisation'):
            contracted = tf.TensorArray(tf.float32, size=length,
                                        element_shape=tf.TensorShape([None, None, None]),
                                        clear_after_read=False)
            _, contracted, _ = tf.while_loop(cond=cond, body=self.RMPS._contract_input_with_nodes,
                                                   loop_vars=[0, contracted, feature],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                                    tf.TensorShape(None)])
            C1s = tf.TensorArray(tf.float32, size=length, element_shape=tf.TensorShape([None, None]), clear_after_read=False)
            C2s = tf.TensorArray(tf.float32, size=length, element_shape=tf.TensorShape([None, None, None]), clear_after_read=False)
            C1_0 = tf.einsum('i,tij->tj', self.RMPS.w_zero, contracted.read(0))
            C1s = C1s.write(0, C1_0)
            C2_l = tf.einsum('tij,jl->til', contracted.read(length - 1), self.RMPS.w_final)
            C2s = C2s.write(length-1, C2_l)
            _, C1s, _, _ = tf.while_loop(cond=cond, body=self._contract_chain_l,
                                                   loop_vars=[1, C1s, contracted, C1_0],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                                    tf.TensorShape(None), tf.TensorShape([None, None])])
            _, C2s, _, _ = tf.while_loop(cond=condr, body=self._contract_chain_r,
                                                   loop_vars=[length-2, C2s, contracted, C2_l],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                                    tf.TensorShape(None), tf.TensorShape([None, None, None])])
        self.C1s = C1s
        self.C2s = C2s
        self.predictions = tf.einsum('til,i->tl', self.C2s.read(0), self.RMPS.w_zero)
        self.dc_dfl = self.predictions - labels
        self.dc_dfl.set_shape([None, None])

    def train_step(self, features, labels):
        self._update_w(features)
        self._update_w_zero()
        self._update_w_final()
        self.RMPS.w = self._updated_w
        self.RMPS.w_zero = self._updated_w_zero
        self.RMPS.w_final = self._updated_w_final


    def _update_w(self, features):
        cond = lambda c, *args: tf.less(c, self.length - 1)
        initial_right = self.C2s.read(1)
        initial_right.set_shape([None, None, None])
        initial_left = self.RMPS.w_zero
        initial_left.set_shape([None])
        right_with_dc_dfl = tf.einsum('til,tl->ti', initial_right, self.dc_dfl)
        combined_l_feature = tf.einsum('j,tn->tjn', initial_left, features[0])
        initial_gradient = tf.einsum('tj,tin->nij', right_with_dc_dfl, combined_l_feature)
        _, gradient, _ = tf.while_loop(cond=cond, body=self._get_gradient_for_w,
                                                   loop_vars=[1, initial_gradient, features],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None, None]),
                                                                    tf.TensorShape([None, None, None])])
        final_right = self.RMPS.w_final
        final_left = self.C1s.read(self.length - 1)
        right_with_dc_dfl = tf.einsum('il,tl->ti', final_right, self.dc_dfl)
        combined_l_feature = tf.einsum('tj,tn->tjn', final_left, features[self.length - 1])
        combined_all = tf.einsum('tj,tin->nij', right_with_dc_dfl, combined_l_feature)
        gradient = tf.add(gradient, combined_all) * self.rate_of_change
        updated_w = tf.add(self.RMPS.w, gradient)
        dims = tf.shape(updated_w)
        l_dim = dims[0] * dims[1]
        r_dim = dims[2]
        flattened_updated_w = tf.reshape(updated_w, [l_dim, r_dim])
        s, u, v = tf.svd(flattened_updated_w)
        filtered_u = utils.check_nan(u, 'u', replace_nan=True)
        filtered_v = utils.check_nan(v, 'v', replace_nan=True)

        filtered_s = tf.boolean_mask(s, tf.greater(s, self.min_singular_value))
        s_size = tf.size(filtered_s)
        # s_size = tf.Print(s_size, [s_size], message='bond dim: ')
        # TODO: Have min_Size settable
        min_size = 3
        case1 = lambda: min_size
        case2 = lambda: self.max_size
        case3 = lambda: s_size
        m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, self.max_size): case2}, default=case3,
                    exclusive=True)
        u_cropped = filtered_u[:, 0:m]
        u_cropped = tf.reshape(u_cropped, [dims[0], m, m])
        v_cropped = tf.transpose(filtered_v[:, 0:m])

        self.v_matrix = v_cropped
        self._updated_w = tf.einsum('ij,jnl->inl', v_cropped, u_cropped)


    def _get_gradient_for_w(self, counter, gradient, features):
        left = self.C1s.read(counter-1)
        right = self.C2s.read(counter+1)
        right_with_dc_dfl = tf.einsum('til,tl->ti', right, self.dc_dfl)
        combined_l_feature = tf.einsum('tj,tn->tjn', left, features[counter])
        combined_all = tf.einsum('tj,tin->nij', right_with_dc_dfl, combined_l_feature)
        gradient = tf.add(gradient, combined_all)
        return [counter + 1, gradient, features]

    def _update_w_zero(self):
        gradient = tf.einsum('til,tl->i', self.C2s.read(0), self.dc_dfl)
        gradient = gradient * self.rate_of_change
        _updated_w_zero = tf.add(gradient, self.RMPS.w_zero)
        self._updated_w_zero = tf.tensordot(_updated_w_zero, self.v_matrix, [[0], [0]])

    def _update_w_final(self):
        gradient = tf.einsum('tj,tl->jl', self.C1s.read(self.length - 1), self.dc_dfl)
        gradient = gradient * self.rate_of_change
        _updated_w_final = tf.add(gradient, self.RMPS.w_final)
        self._updated_w_final = tf.tensordot(self.v_matrix, _updated_w_final, [[0], [1]])

    def _contract_chain_l(self, counter, contracted_chain, contracted, previous):
        contracted_tensor = contracted.read(counter)
        contracted_tensor.set_shape([None, None, None])
        previous = tf.einsum('ti,tij->tj', previous, contracted_tensor)
        contracted_chain = contracted_chain.write(counter, previous)
        return [counter + 1, contracted_chain, contracted, previous]

    def _contract_chain_r(self, counter, contracted_chain, contracted, previous):
        contracted_tensor = contracted.read(counter)
        contracted_tensor.set_shape([None, None, None])
        previous = tf.einsum('tij,tjl->til', contracted_tensor, previous)
        contracted_chain = contracted_chain.write(counter, previous)
        return [counter - 1, contracted_chain, contracted, previous]

    def _test_step(self, feature, label):
        """
        A single step of training
        :param self:
        :param feature:
        :param label:
        :return:
        """
        f = self.RMPS.predict(feature)
        cost = self.RMPS.cost(f, label)
        accuracy = self.RMPS.accuracy(f, label)
        confusion = self.RMPS.confusion_matrix(f, label)
        return cost, accuracy, confusion

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
        optimised_w = None
        optimised_w_zero = None
        optimised_w_final = None
        lr = optional_parameters.rate_of_change
        train_cost, train_accuracy, train_confusion = self._test_step(self._feature, self._label)
        actual_length, (test_feature, test_label) = data_source.test_data_of_length(self.current_length)
        print("Length of test data:", actual_length)

        feature = tf.placeholder(tf.float32, shape=[None, None, self.RMPS.d_feature])
        label = tf.placeholder(tf.float32, shape=[None, self.RMPS.d_output])
        test_cost, test_accuracy, test_confusion = self._test_step(feature, label)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_step):
                start = time.time()
                if length is not None:
                    self.current_length = length(i)
                actual_length, (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size, self.current_length)
                print("Length of training data:", actual_length)
                self.feed_dict = self.RMPS.create_feed_dict(optimised_w_zero, optimised_w, optimised_w_final)
                self.feed_dict[self._feature] = batch_feature
                self.feed_dict[self._label] = batch_label
                self.feed_dict[self.rate_of_change] = lr
                self.feed_dict[feature] = test_feature
                self.feed_dict[label] = test_label
                to_eval = [train_cost, train_accuracy, test_cost, test_accuracy, test_confusion, self.RMPS.w, self.RMPS.w_zero, self.RMPS.w_final]
                train_c, train_acc, test_c, test_acc, test_conf, optimised_w, optimised_w_zero, optimised_w_final = sess.run(to_eval,
                                                                                                                  feed_dict=self.feed_dict,
                                                                                                                  options=run_options,
                                                                                                                  run_metadata=run_metadata)
                end = time.time()
                print(time.strftime("%Y-%m-%d %H:%M:%S"))
                print('step {}, training cost {}, accuracy {}. Took {} s'.format(i, train_c, train_acc, end - start))
                print('step {}, testing cost {}, accuracy {}'.format(i, test_c, test_acc))
                print('confusion matrix: \n' + str(test_conf))





