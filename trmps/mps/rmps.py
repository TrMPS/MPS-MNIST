import tensorflow as tf
import numpy as np

class RMPS(object):
    def __init__(self, expected_length, d_feature, d_output):
        self.d_feature = d_feature
        self.d_output = d_output
        self._rand_initalise(expected_length)

    def _rand_initalise(self, expected_length):
        initial_bond_dim = 2
        _w_zero = np.zeros([initial_bond_dim], dtype=np.float32)
        _w_zero[0] = 1.0
        self.w_zero = tf.placeholder_with_default(tf.Variable(_w_zero, dtype=tf.float32, name='w_zero'),
                                                  [None])
        _w = np.tile(np.array([[1.0, 0.0],[0.0, 1.0]], dtype=np.float32) / (expected_length*self.d_feature), (self.d_feature, 1, 1))
        self.w = tf.placeholder_with_default(tf.Variable(_w, dtype=tf.float32, name='w'),
                                             [self.d_feature, None, None])
        _w_final = np.tile(_w_zero, (self.d_output, 1))
        _w_final = np.transpose(_w_final)
        print(_w_final.shape)
        print(_w_final)
        self.w_final = tf.placeholder_with_default(tf.Variable(_w_final, dtype=tf.float32, name='w_final'),
                                      [None, self.d_output])

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
        length = tf.shape(feature)[0]
        cond = lambda c, *args: tf.less(c, length)
        with tf.name_scope('calculate_contracted'):
            contracted = tf.TensorArray(tf.float32, size=length,
                                        element_shape=tf.TensorShape([None, None, None]))
            _, contracted, _ = tf.while_loop(cond=cond, body=self._contract_input_with_nodes,
                                                   loop_vars=[0, contracted, feature],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                                    tf.TensorShape(None)])
        with tf.name_scope('multiply_chain'):
            contracted_chain = tf.tensordot(self.w_zero, contracted.read(0), [[0], [1]])
            _, contracted_chain, _ = tf.while_loop(cond=cond, body=self._contract_chain,
                                                   loop_vars=[1, contracted_chain, contracted],
                                                   shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                                    tf.TensorShape(None)])
        with tf.name_scope('multiply_last'):
            prediction = tf.tensordot(contracted_chain, self.w_final, [[1], [0]])
        return prediction

    def _contract_input_with_nodes(self, counter, contracted, feature):
        contracted = contracted.write(counter, tf.tensordot(feature[counter], self.w, [[1], [0]]))
        return [counter+1, contracted, feature]

    def _contract_chain(self, counter, contracted_chain, contracted):
        contracted_tensor = contracted.read(counter)
        contracted_chain = tf.einsum('ti,tij->tj', contracted_chain, contracted_tensor)
        return [counter+1, contracted_chain, contracted]

    def create_feed_dict(self, w_zero, w, w_final):
        # When feeding in, make sure all of the above values are either None or not None.
        feed_dict = {}
        if w_zero is not None:
            feed_dict[self.w_zero] = w_zero
            feed_dict[self.w] = w
            feed_dict[self.w_final] = w_final
        return feed_dict

if __name__ =='__main__':
    test = RMPS(2, 2, 10)
    feature = np.array([[0.0,1.0],[0.0,1.0]], dtype=np.float32)
    prediction = test.predict(feature)
