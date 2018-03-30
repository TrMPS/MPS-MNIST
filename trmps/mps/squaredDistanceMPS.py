from mps.mps import MPS
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# class sqMPSOptimizer(MPSOptimizer):
#     """
#     A subclass of MPSOptimizer that uses the square error instead of the cross entropy.
#     Should be used in the exact same way, but with sqMPS instead of MPS.
#     """

#     def _calculate_hessian(self, f, C):
#         with tf.name_scope('hessian'):
#             d1 = tf.shape(C)[-2]
#             d2 = tf.shape(C)[-1]
#             C_sq = tf.square(C)
#             hessian = C_sq+ 2 * self.reg
#             hessian = tf.reduce_sum(hessian, axis = 0)
#             hessian = tf.expand_dims(hessian, axis = 0)

#             return hessian
