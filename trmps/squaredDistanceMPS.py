from mps import MPS
from optimizer import MPSOptimizer
import tensorflow as tf

class sqMPS(MPS):
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

    def _update_bond(self, bond, C, acc_lr_reg, counter):
        # obtain the original cost
        # bond = tf.Print(bond, [counter, tf.shape(bond)])
        f, cost = self._get_f_and_cost(bond, C)
        # h = self._calculate_hessian(f, C)
        h = 1.0

        # perform gradient descent on the bond
        with tf.name_scope("tensordotgradient"):
            gradient = tf.tensordot(self._label - f, C, [[0], [0]]) - 2 * self.reg * bond
            delta_bond = gradient / h
        gradient_dot_change = tf.tensordot(gradient,
                                           delta_bond,
                                           [[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]])/tf.cast(self.batch_size, tf.float32)
        lr = self.rate_of_change
        lr, updated_bond = self._armijo_loop(bond, C, lr, cost, delta_bond, gradient_dot_change)

        _, cost1 = self._get_f_and_cost(updated_bond, C)
        if self.verbosity != 0:
            updated_bond = tf.Print(updated_bond, [cost1], message='updated cost', first_n=self.verbosity)
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond
