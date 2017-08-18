from mps import MPS
from optimizer import MPSOptimizer
import tensorflow as tf

# TODO: Deprecate everything in this file and move these things into some sort of configuration for the
#       MPS and MPSOptimizer classes

class sqMPS(MPS):
    """
    A subclass of MPS that uses the square error instead of the cross entropy.
    Should be used in the exact same way, but with sqMPSOptimizer instead of MPSOptimizer.
    """
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

    def _cost_for_lin_reg(self, labels, predictions):
        return 0.5 * tf.reduce_sum(tf.square(predictions-labels))

class sqMPSOptimizer(MPSOptimizer):
    """
    A subclass of MPSOptimizer that uses the square error instead of the cross entropy.
    Should be used in the exact same way, but with sqMPS instead of MPS.
    """
    def _get_f_and_cost(self, bond, C):
        """

        :param self:
        :param bond:
        :param C:
        :return:
        """
        with tf.name_scope("tensordotf"):
            # f = tf.einsum('lmnik,tmnik->tl', bond, C)
            f = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
            # f = tf.nn.softmax(f)
        with tf.name_scope("reduce_sumcost"):
            cost = 0.5 * tf.reduce_mean(tf.square(f-self._label))

        return f, cost

    def _calculate_hessian(self, f, C):
        with tf.name_scope('hessian'):
            d1 = tf.shape(C)[-2]
            d2 = tf.shape(C)[-1]
            C_sq = tf.square(C)
            hessian = C_sq+ 2 * self.reg
            hessian = tf.reduce_sum(hessian, axis = 0)
            hessian = tf.expand_dims(hessian, axis = 0)

            return hessian
