import tensorflow as tf
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils

class UMPS(object):
    def __init__(self, expected_length, d_feature, centre_loc=10, initial_bond_dim=2):
        self.d_feature = d_feature
        self.centre_loc = centre_loc
        self._rand_initalise(expected_length, initial_bond_dim)

    def _rand_initalise(self, expected_length, initial_bond_dim=2):
        if (initial_bond_dim / 2.0) - np.floor(initial_bond_dim / 2.0) != 0.0:
            raise ValueError('Initial bond dim should be an even number')
        _w_c = np.full((initial_bond_dim, initial_bond_dim), 0.0).astype(np.float32)
        _w_c = np.tile(_w_c, (self.d_feature, 1, 1))
        self.weight_centre = tf.placeholder_with_default(_w_c, [self.d_feature, None, None])
        print(_w_c.shape)
        _c = np.identity(initial_bond_dim).astype(np.float32)
        print(_c.shape)
        self._c = tf.placeholder_with_default(_c, [None, None])

        _left = np.einsum('mij,kj->mik', _w_c, _c)
        _left_dims = _left.shape
        print(_left_dims)
        _left_flattened = np.reshape(_left, [_left_dims[0] * _left_dims[1], _left_dims[2]])
        print(_left_flattened.shape)
        u, s, v = np.linalg.svd(_left_flattened)
        # In the case of NumPy, the value of V returned is actually V^dagger, so don't need to worry about transposing
        filtered_u = u[:, :v.shape[1]]
        filtered_v = v
        uv = np.matmul(filtered_u, filtered_v)
        print(uv.shape)
        _w_l = np.reshape(uv, [_left_dims[0], _left_dims[1], _left_dims[2]])
        self.weight_left = tf.placeholder_with_default(_w_l, [self.d_feature, None, None])
        print(_c.shape)

        _right = np.einsum('ik,mij->mkj', _c, _w_c)
        _right_dims = _right.shape
        _right_flattened = np.reshape(_right, [_right_dims[0] * _right_dims[1], _right_dims[2]])
        u, s, v = np.linalg.svd(_right_flattened)
        filtered_u = u[:, :v.shape[1]]
        filtered_v = v
        uv = np.matmul(filtered_u, filtered_v)
        _w_r = np.reshape(uv, [_right_dims[0], _right_dims[1], _right_dims[2]])
        self.weight_right = tf.placeholder_with_default(_w_r, [self.d_feature, None, None])
        print(_w_r)
        print(_w_l)


    def cost(self, f, labels):
        """
        Returns the cost (softmax cross entropy with logits) given the predictions(f), and the correct labels
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow scalar
            The cost of the predictions, as judged by softmax cross entropy with logits
        """
        with tf.name_scope("cost"):
            cost = 0.5 * tf.reduce_mean(tf.square(f - labels))
        return cost

    def accuracy(self, f, labels):
        """
        Returns the accuracy given the predictions(f), and the correct labels.
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow scalar
            The accuracy of the predictions
        """
        # with tf.name_scope("accuracy"):
        #     prediction = tf.argmax(f, axis=1)
        #     true_value = tf.argmax(labels, axis=1)
        #     correct_prediction = tf.equal(prediction, true_value)
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.constant(1.0)
        return accuracy

    def confusion_matrix(self, f, labels):
        # TODO: add a normalisation option.
        """
        Returns the confusion matrix given the predictions(f), and the correct labels
        :param f: tensorflow Tensor of shape (batch_size, d_output)
            The predictions that are to be judged.
        :param labels: tensorflow Tensor of shape (batch_size, d_output)
            The correct labels.
        :return: a tensorflow Tensor of shape (d_output, d_output)
            The confusion matrix
        """
        predictions = tf.argmax(f, axis=1)
        true_values = tf.argmax(labels, axis=1)
        confusion_mat = tf.confusion_matrix(
            true_values, predictions, num_classes=self.d_output)
        return confusion_mat

    def predict(self, feature):
        self.feature = feature
        length = tf.shape(feature)[0]
        with tf.name_scope('calculate_C1'):

            C1 = tf.tensordot(self.feature[0], self.weight_left, [[1], [0]])
            cond = lambda c, b: tf.less(c, self.centre_loc)
            counter, C1 = tf.while_loop(cond=cond, body=self._chain_multiply_r, loop_vars=[1, C1],
                                        shape_invariants=[tf.TensorShape(
                                            []), tf.TensorShape([None, None, None])],
                                        parallel_iterations=5)

        with tf.name_scope('calculate_C2'):
            C2 = tf.tensordot(self.feature[-1], self.weight_right, [[1], [0]])
            cond2 = lambda c, b: tf.greater(c, self.centre_loc)
            _, C2 = tf.while_loop(cond=cond2, body=self._chain_multiply_l, loop_vars=[length - 1, C2],
                                  shape_invariants=[tf.TensorShape(
                                      []), tf.TensorShape([None, None, None])],
                                  parallel_iterations=5)
        contracted_sp_node = tf.einsum(
            'nij,tn->tij', self.weight_centre, feature[self.centre_loc])
        C1_sp = tf.einsum('tij,tjk->tik', C1, contracted_sp_node)
        all_contracted = tf.einsum('tij,tjk->tik', C1_sp, C2)
        predictions = tf.einsum('tij,tij->t', all_contracted, all_contracted)
        return predictions

    def _chain_multiply_l(self, counter, C2):
        with tf.name_scope('chain_multiply_left'):
            node = self.weight_right
            contracted_node = tf.tensordot(self.feature[counter], node,
                                           [[1], [0]])
            # C2 = tf.Print(C2, [C2])
            C2 = tf.einsum('tij,tjl->til', contracted_node, C2)
            counter = counter - 1

        return [counter, C2]

    def _chain_multiply_r(self, counter, C1):
        with tf.name_scope('chain_multiply_right'):
            node = self.weight_left
            contracted_node = tf.tensordot(self.feature[counter], node,
                                           [[1], [0]])
            # C1 = tf.Print(C1, [C1])
            C1 = tf.einsum('tli,tij->tlj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1]

    def create_feed_dict(self, weight_centre, c, weight_left, weight_right):
        # When feeding in, make sure all of the above values are either None or not None.
        feed_dict = {}
        if weight_left is not None:
            feed_dict[self.weight_left] = weight_left
            feed_dict[self.weight_centre] = weight_centre
            feed_dict[self.weight_right] = weight_right
            feed_dict[self._c] = c
        return feed_dict

if __name__ == '__main__':
    test = UMPS(2, 5)
