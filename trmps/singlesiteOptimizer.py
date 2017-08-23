from baseoptimizer import *

class SingleSiteMPSOptimizer(BaseOptimizer):

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
            cond = lambda counter, *args: tf.greater(counter, special_loc)
            _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[self.MPS.input_size-1, C2, C2s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC2")

    def _update_left(self, counter, acc_lr_reg, C2s, updated_nodes, previous_node):

        with tf.name_scope("update_left"):
            # Read in the nodes
            new_node = self.MPS.nodes.read(counter - 1)
            new_node.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = C2s.read(counter)
            C1 = self.C1s.read(counter)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input = self._feature[counter]

            # Calculate the bond
            bond = previous_node
            bond = tf.transpose(bond, [0, 1, 3, 2])

            C = self._calculate_C(C2, C1, input)

            # update the bond
            updated_bond = self._update_bond(bond, C)


            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            aj = tf.transpose(aj, perm=[0, 2, 1])
            aj1 = tf.transpose(aj1, perm=[2, 1, 0])
            new_node = tf.tensordot(aj1, new_node,[[0], [2]])
            new_node = tf.transpose(new_node, [0, 2, 3, 1])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)
            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC2"):
                C2 = tf.einsum('tij,tj->ti', contracted_aj, C2)
            C2s = C2s.write(counter - 1, C2)
            # counter = tf.Print(counter, [counter])
            updated_counter = counter - 1
            acc_lr_reg = acc_lr_reg * self.lr_reg

        return [updated_counter, acc_lr_reg, C2s, updated_nodes, new_node]

    def _update_right(self, counter, acc_lr_reg, C1s, updated_nodes, previous_node):
        with tf.name_scope("update_right"):
            # Read in the nodes
            new_node = self.MPS.nodes.read(counter + 1)
            new_node.set_shape([self.MPS.d_feature, None, None])

            # Calculate the C matrix
            C2 = self.C2s.read(counter)
            C1 = C1s.read(counter)
            C1.set_shape([None, None])
            C2.set_shape([None, None])
            input = self._feature[counter]

            # Calculate the bond
            bond = previous_node
            # bond = tf.transpose(tf.tensordot(n1, n2, [[3],[1]]), [0, 1, 3, 2, 4])
            # einsum is actually faster in this case

            C = self._calculate_C(C1, C2, input)

            # Update the bond
            updated_bond = self._update_bond(bond, C)

            # Decompose the bond
            aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
            new_node = tf.tensordot(aj1, new_node,[[2], [1]])
            new_node = tf.transpose(new_node, [1, 2, 0, 3])

            # Transpose the values and add to the new variables
            updated_nodes = updated_nodes.write(counter, aj)

            with tf.name_scope("tensordotcontracted_aj"):
                # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                contracted_aj = tf.tensordot(self._feature[counter], aj, [[1], [0]])
            with tf.name_scope("einsumC1"):
                C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
            C1s = C1s.write(counter+1, C1)
            # counter = tf.Print(counter, [counter])
            updated_counter = counter + 1
            acc_lr_reg = acc_lr_reg * self.lr_reg

        return [updated_counter, acc_lr_reg, C1s, updated_nodes, new_node]

    def _calculate_C(self, C1, C2, input):
        # C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, input1, input2)
        # C = tf.einsum('ti,tk,tm->tmik', C1, C2, input)
        d1 = tf.shape(C1)[1]
        d2 = tf.shape(C2)[1]

        with tf.name_scope("calculateC"):
            C1 = tf.reshape(C1, [self.batch_size, 1, d1, 1])
            C2 = tf.reshape(C2, [self.batch_size, 1, 1, d2])
            input = tf.reshape(input, [self.batch_size, self.MPS.d_feature, 1, 1])
            C = C1 * C2 * input

        return C

    def _get_f_and_h(self, bond, C):
        with tf.name_scope("tensordotf"):
            # f = tf.einsum('lmik,tmik->tl', bond, C)
            f = tf.tensordot(C, bond, [[1, 2, 3], [1, 2, 3]])
            h = tf.nn.softmax(f)
        return f, h

    def _calculate_hessian(self, f, C):
        # TODO: IMPLEMENT
        print("WARNING: Calculating the Hessian for Single Site DMRG is not yet implemented; this will probably cause your script to not work/ crash")
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
                                           [[0, 1, 2, 3], [0, 1, 2, 3]])/tf.cast(self.batch_size, tf.float32)
        lr = self.rate_of_change
        lr, updated_bond = self._armijo_loop(bond, C, lr, cost, delta_bond, gradient_dot_change)

        _, cost1 = self._get_f_and_cost(updated_bond, C)
        if self.verbosity != 0:
            updated_bond = tf.Print(updated_bond, [cost1], message='updated cost', first_n=self.verbosity)
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond

    def _bond_decomposition(self, bond, max_size, min_size=3):
        with tf.name_scope("bond_decomposition"):
            bond_reshaped = tf.transpose(bond, perm=[1, 2, 0, 3])
            # bond_reshaped = tf.Print(bond_reshaped, [tf.shape(bond_reshaped), tf.shape(bond)], summarize = 1000, message = "bond reshaped, bond")

            dims = tf.shape(bond_reshaped)
            l_dim = dims[0] * dims[1]
            r_dim = dims[2] * dims[3]
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
            a_prime_j1 = tf.reshape(sv, [m, dims[2], dims[3]])
            # a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])
            # will do this in the update_right/update_left functions from now on as else transpose twice for udpate_left

        return (a_prime_j, a_prime_j1)

