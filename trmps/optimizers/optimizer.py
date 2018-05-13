import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers.baseoptimizer import *

class MPSOptimizer(BaseOptimizer):
    """
    MPSOptimizer is used to optimize the MPS class for a dataset.
    Implements the DMRG method, as detailed in the paper
    Supervised learning with Quantum-Inspired Tensor Networks
    by E.Miles Stoudenmire and David J.Schwab
    Example usage using the MNIST dataset:

    from optimizer import *
    import MNISTpreprocessing

    # Model parameters
    d_feature = 2
    d_output = 10
    batch_size = 1000
    permuted = False
    shuffled = False
    shrink = True
    input_size = 784
    if shrink:
        input_size = 196

    max_size = 20

    rate_of_change = 10 ** (-7)
    logging_enabled = False

    cutoff = 10 # change this next
    n_step = 10

    data_source = MNISTpreprocessing.MNISTDatasource(shrink = shrink, permuted = permuted, shuffled = shuffled)

    weights = None

    network = MPS(d_feature, d_output, input_size)
    network.prepare(data_source)
    optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
    optimizer.train(data_source, batch_size, n_step,
                    rate_of_change=rate_of_change,
                    logging_enabled=logging_enabled,
                    initial_weights=weights)
    """

    def _setup_optimization(self):
        """
        :param self:
        :return:
        """
        '''
        C1s: size = input_size (as the last one is kept redundant)
        C2s: size = input_size (first one redundant)

        writes C1 from 0 to special_loc-1
        writes C2 from special_loc to size-3 (corresponds to special_loc+2 to size-1 the nodes)
        '''
        feature = self._feature
        nodes = self.MPS.nodes
        special_loc = self.MPS._special_node_loc
        batch_size = tf.shape(feature)[1]

        with tf.name_scope("setup_optimization"):


            C1 = tf.tile(self.MPS.start_node, [batch_size, 1])
            C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
            C1s = C1s.write(0, C1)
            cond = lambda c, *args: tf.less(c, special_loc)
            _, _, self.C1s = tf.while_loop(cond=cond, body=self._find_C1, loop_vars=[0, C1, C1s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC1")

            C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=False, clear_after_read=False)
            C2 = tf.tile(self.MPS.end_node, [batch_size, 1])
            C2s = C2s.write(self.MPS.input_size-1, C2)
            cond = lambda counter, *args: tf.greater(counter, special_loc+1)
            _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[self.MPS.input_size-1, C2, C2s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC2")

    def _update_left(self, counter, acc_lr_reg, C2s, updated_nodes, previous_node, costs):
        """

        :param self:
        :param counter:
        :param C2s:
        :param updated_nodes:
        :param previous_node:
        :return:
        """

        with tf.name_scope("update_left"):
            # Read in the nodes
            n1 = previous_node
            n2 = self.MPS.nodes.read(counter - 1)
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = C2s.read(counter)
            C1 = self.C1s.read(counter - 1)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input1 = self._feature[counter - 1]
            input2 = self._feature[counter]

            # Calculate the bond
            bond = tf.einsum('nkj,lmji->lmnik', n2, n1)

            C = self._calculate_C(C2, C1, input2, input1)

            # update the bond
            updated_bond, cost = self._repeatedly_update_bond(bond, C)
            costs = costs.write(counter - 1, cost)


            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            aj = tf.transpose(aj, perm=[0, 2, 1])
            aj1 = tf.transpose(aj1, perm=[1, 2, 3, 0])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)
            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC2"):
                C2 = tf.einsum('tij,tj->ti', contracted_aj, C2)
            C2s = C2s.write(counter - 1, C2)
            #counter = tf.Print(counter, [counter])
            updated_counter = counter - 1
            acc_lr_reg = acc_lr_reg * self.lr_reg



        return [updated_counter, acc_lr_reg, C2s, updated_nodes, aj1, costs]

    def _update_right(self, counter, acc_lr_reg, C1s, updated_nodes, previous_node, costs):
        """

        :param self:
        :param counter:
        :param C1s:
        :param updated_nodes:
        :param previous_node:
        :return:
        """
        with tf.name_scope("update_right"):
            # Read in the nodes
            n1 = previous_node
            n2 = self.MPS.nodes.read(counter + 1)
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = self.C2s.read(counter + 1)
            C1 = C1s.read(counter)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input1 = self._feature[counter]
            input2 = self._feature[counter + 1]

            # Calculate the bond
            bond = tf.einsum('lmij,njk->lmnik', n1, n2)
            # bond = tf.transpose(tf.tensordot(n1, n2, [[3],[1]]), [0, 1, 3, 2, 4])
            # einsum is actually faster in this case

            C = self._calculate_C(C1, C2, input1, input2)

            # Update the bond
            updated_bond, cost = self._repeatedly_update_bond(bond, C)
            costs = costs.write(counter, cost)

            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            aj1 = tf.transpose(aj1, perm=[1, 2, 0, 3])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)

            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC1"):
                C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
            C1s = C1s.write(counter+1, C1)
            #counter = tf.Print(counter, [counter])
            updated_counter = counter + 1
            acc_lr_reg = acc_lr_reg * self.lr_reg

        return [updated_counter, acc_lr_reg, C1s, updated_nodes, aj1, costs]

    def _calculate_C(self, C1, C2, input1, input2):
        """

        :param self:
        :param C1:
        :param C2:
        :param input1:
        :param input2:
        :return:
        """
        # C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, input1, input2)
        d1 = tf.shape(C1)[1]
        d2 = tf.shape(C2)[1]

        with tf.name_scope("calculateC"):
            C1 = tf.reshape(C1, [self.batch_size, 1, 1, d1, 1])
            C2 = tf.reshape(C2, [self.batch_size, 1, 1, 1, d2])
            input1 = tf.reshape(input1, [self.batch_size, self.MPS.d_feature, 1, 1, 1])
            input2 = tf.reshape(input2, [self.batch_size, 1, self.MPS.d_feature, 1, 1])
            intermediate_1 = C1 * C2
            intermediate_2 = input1 * input2
            C = intermediate_1 * intermediate_2

        return C

    def _get_f_and_h(self, bond, C):
        with tf.name_scope("tensordotf"):
            # f = tf.einsum('lmnik,tmnik->tl', bond, C)
            f = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
            h = tf.nn.softmax(f)
        return f, h

    def _calculate_hessian(self, f, C):
        with tf.name_scope('hessian'):
            d1 = tf.shape(C)[-2]
            d2 = tf.shape(C)[-1]
            f_part = tf.reshape(f * (1 - f), [self.batch_size, self.MPS.d_output, 1, 1, 1, 1])
            C_sq = tf.reshape(tf.square(C), [self.batch_size, 1, self.MPS.d_feature, self.MPS.d_feature, d1, d2])
            hessian = tf.reduce_sum(f_part * C_sq, axis=0) + 2 * self.reg

            return hessian

    def _update_bond(self, bond, C):
        # obtain the original cost
        # bond = tf.Print(bond, [counter, tf.shape(bond)])
        f, cost = self._get_f_and_cost(bond, C)
        h = 1.0
        if self.use_hessian:
            h = self._calculate_hessian(f, C)

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

        return updated_bond, cost

    def _bond_decomposition(self, bond, max_size, min_size=3):
        """

        :param self:
        :param bond:
        :param max_size:
        :param min_size:
        :param min_singular_value:
        :return:
        """
        """
        Decomposes bond, so that the next step can be done.
        :param bond:
        :param m:
        :return:
        """
        with tf.name_scope("bond_decomposition"):
            bond_reshaped = tf.transpose(bond, perm=[1, 3, 0, 2, 4])
            # bond_reshaped = tf.Print(bond_reshaped, [tf.shape(bond_reshaped), tf.shape(bond)], summarize = 1000, message = "bond reshaped, bond")

            dims = tf.shape(bond_reshaped)
            l_dim = dims[0] * dims[1]
            r_dim = dims[2] * dims[3] * dims[4]
            bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
            s, u, v = tf.svd(bond_flattened)
            filtered_u = utils.check_nan(u, 'u', replace_nan=True)
            filtered_v = utils.check_nan(v, 'v', replace_nan=True)

            filtered_s = tf.boolean_mask(s, tf.greater(s, self.min_singular_value))
            s_size = tf.size(filtered_s)
            # s_size = tf.Print(s_size, [s_size], message='bond dim: ')

            case1 = lambda: min_size
            case2 = lambda: max_size
            case3 = lambda: s_size
            m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3,
                        exclusive=True)
            # if self.verbosity != 0:
            #     m = tf.Print(m, [m, s[m-5:m]], first_n=self.verbosity, summarize=5, message='bond: ')

            # make s into a matrix
            s_mat = tf.diag(s[0:m])

            # make u, v into suitable matrices
            u_cropped = filtered_u[:, 0:m]
            v_cropped = tf.transpose(filtered_v[:, 0:m])

            # make a_
            a_prime_j = tf.reshape(u_cropped, [dims[0], dims[1], m])

            sv = tf.matmul(s_mat, v_cropped)
            a_prime_j1 = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
            # a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])
            # will do this in the update_right/update_left functions from now on as else transpose twice for udpate_left

        return (a_prime_j, a_prime_j1)
