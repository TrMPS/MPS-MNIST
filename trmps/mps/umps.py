import tensorflow as tf
import numpy as np

class UMPS(object):
    def __init__(self, expected_length, d_feature, d_output, centre_loc=10, initial_bond_dim=2):
        self.d_feature = d_feature
        self.centre_loc = centre_loc
        self.d_output = d_output
        self._rand_initalise(expected_length, initial_bond_dim)

    def _rand_initalise(self, expected_length, initial_bond_dim=2):
        if (initial_bond_dim / 2.0) - np.floor(initial_bond_dim / 2.0) != 0.0:
            raise ValueError('Initial bond dim should be an even number')
        _w_c = np.identity(initial_bond_dim)
        _w_c = np.tile(_w_c, (self.d_output, self.d_feature, 1, 1))
        self.weight_centre = _w_c


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
            cost = 0.5 * tf.reduce_mean(tf.square(f-labels))
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
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(f, axis=1)
            true_value = tf.argmax(labels, axis=1)
            correct_prediction = tf.equal(prediction, true_value)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
            _, C2 = tf.while_loop(cond=cond2, body=self._chain_multiply_l, loop_vars=[self.input_size - 1, C2],
                                  shape_invariants=[tf.TensorShape(
                                      []), tf.TensorShape([None, None, None])],
                                  parallel_iterations=5)
        contracted_sp_node = tf.einsum(
            'lnij,tn->tlij', self.weight_centre, feature[self.centre_loc])
        C1_sp = tf.einsum('tij,tljk->tlik', C1, contracted_sp_node)
        all_contracted = tf.einsum('tlij,tjk->tlik', C1_sp, C2)
        predictions = tf.einsum('tlij,tlij->tl', all_contracted, all_contracted)
        return predictions



    def _chain_multiply_l(self, counter, C2):
        with tf.name_scope('chain_multiply_left'):
            node = self.weight_right
            contracted_node = tf.tensordot(self.feature[counter], node,
                                           [[1], [0]])
            C2 = tf.einsum('tij,tjl->til', contracted_node, C2)
            counter = counter - 1

        return [counter, C2]

    def _chain_multiply_r(self, counter, C1):
        with tf.name_scope('chain_multiply_right'):
            node = self.weight_left
            contracted_node = tf.tensordot(self.feature[counter], node,
                                           [[1], [0]])
            C1 = tf.einsum('tli,tij->tlj', C1, contracted_node)
            counter = counter + 1
        return [counter, C1]

    def create_feed_dict(self, weight_left, weight_right, weight_centre):
        # When feeding in, make sure all of the above values are either None or not None.
        feed_dict = {}
        if w_zero is not None:
            feed_dict[self.weight_left] = weight_left
            feed_dict[self.weight_centre] = weight_centre
            feed_dict[self.weight_right] = weight_right
        return feed_dict

if __name__ == '__main__':
    test = UMPS(2, 3, 5)
