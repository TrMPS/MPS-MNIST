from optimizer import MPSOptimizer
import tensorflow as tf

class generatorOptimizer(MPSOptimizer):
    def __init__(self, MPSNetwork, max_size, delegate, grad_func=None, cutoff=1000, reg=0.001, min_singular_value=10 ** (-4), verbose=0):
        super().__init__(MPSNetwork, max_size, grad_func, cutoff, reg, min_singular_value, verbose)
        self.generator = self.MPS
        self.delegate = delegate
        self.d_output = self.MPS.d_output
        self._desired_labels = self.delegate._desired_labels
        self._label = tf.tile(delegate.true_label, self.batch_size)
        self._label = tf.reshape(self._desired_labels, [self.batch_size, -1])
        self._phi_part = tf.ones([self.batch_size])
        self._phi_part_special = tf.zeros([self.batch_size])

    def _get_xs(self, bond, C):
        """

        :param self:
        :param bond:
        :param C:
        :return:
        """
        with tf.name_scope("tensordotf"):
            # f = tf.einsum('lmnik,tmnik->tl', bond, C)
            g = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
            h = tf.sigmoid(g)

        return h

    def _update_bond(self, bond, C):
        # obtain the original cost
        xs = self._get_xs(bond, C)
        phis = tf.stack([self._phi_part, xs], axis=1)

        discriminator = self.delegate.discriminator

        # Prepare for finding gradient by getting C1s and C2s for the discriminator network
        full_chain_cond = lambda c, *args: tf.less(c, self.d_output)
        c1_cond = lambda c, *args: tf.less(c, self.d_output - 1)
        greater_than_zero_cond = lambda c, *args: tf.greater(c, 0)
        left_of_special_loc_cond = lambda c, *args: tf.less(c, discriminator._special_node_loc)
        right_of_special_loc_cond = lambda c, *args: tf.greater(c, discriminator._special_node_loc)
        prev_C1 = tf.tile(self.discriminator.start_node, [self.batch_size, 1])
        prev_C2 = tf.tile(self.discriminator.end_node, [self.batch_size, 1])
        C1s = tf.TensorArray(tf.float32, size=self.generator.d_output, infer_shape=False, clear_after_read=False)
        C2s = tf.TensorArray(tf.float32, size=self.generator.d_output, infer_shape=False, clear_after_read=False)
        counter_r, _, _, C1s, prev_C1 = tf.while_loop(cond=left_of_special_loc_cond, body=self._chain_multiply_r,
                                                loop_vars=[0, discriminator.nodes, phis, C1s, prev_C1],
                                                shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                             tf.TensorShape([None, None]), tf.TensorShape(None),
                                                             tf.TensorShape(None)],
                                                parallel_iterations=10,
                                                name="findC1ForGeneratorUpdate")
        counter_l, _, _, C2s, prev_C2 = tf.while_loop(cond=right_of_special_loc_cond, body=self._chain_multiply_l,
                                                loop_vars=[self.generator.d_output-1, discriminator.nodes, phis, C2s, prev_C2],
                                                shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                             tf.TensorShape([None, None]), tf.TensorShape(None),
                                                             tf.TensorShape(None)],
                                                parallel_iterations=10,
                                                name="findC2ForGeneratorUpdate")
        contracted_sp_node = tf.einsum(
                'lnij,tn->tlij', discriminator.nodes.read(discriminator._special_node_loc), phis[discriminator._special_node_loc])
        prev_C1 = tf.einsum('ti,tlij->tlj', prev_C1, contracted_sp_node)
        prev_C2 = tf.einsum('tlij,tj->tli', contracted_sp_node, prev_C2)
        counter_r += 1
        counter_l -= 1
        C1s = C1s.write(counter_r, prev_C1)
        C2s = C2s.write(counter_l, prev_C2)
        counter_r, _, _, C1s, prev_C1 = tf.while_loop(cond=c1_cond, body=self._chain_multiply_r2,
                                                loop_vars=[counter_r, discriminator.nodes, phis, C1s, prev_C1],
                                                shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                             tf.TensorShape([None, None]), tf.TensorShape(None),
                                                             tf.TensorShape(None)],
                                                parallel_iterations=10,
                                                name="findC1ForGeneratorUpdate")
        counter_l, _, _, C2s, prev_C2 = tf.while_loop(cond=greater_than_zero_cond, body=self._chain_multiply_l2,
                                                loop_vars=[counter_l, discriminator.nodes, phis, C2s, prev_C2],
                                                shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                             tf.TensorShape([None, None]), tf.TensorShape(None),
                                                             tf.TensorShape(None)],
                                                parallel_iterations=10,
                                                name="findC2ForGeneratorUpdate")
        last_C2 = C2s.read(self.generator.d_output - 1)
        f = tf.einsum('tli,ti->tl', prev_C1, last_C2)
        f_sub_one = f - 1


        # Prepare for finding the gradient by finding the Aphis for the modified phis
        modified_xs = xs*(xs-1)
        modified_phis = tf.stack([self._phi_part_special, modified_xs], axis=1)
        modified_Aphis = tf.TensorArray(tf.float32, size=self.d_output, clear_after_read=False, infer_shape=False)
        loop_vars = [0, modified_Aphis, discriminator.nodes, modified_phis]
         _, modified_Aphis, _, _ = tf.while_loop(cond=full_chain_cond, body=self._find_Aphi, loop_vars=loop_vars,
                                     shape_invariants=[tf.TensorShape([]), tf.TensorShape(None),
                                                         tf.TensorShape(None), tf.shape(xs)],
                                     parallel_iterations=10,
                                     name = "findModifiedAphis")
        bond_shape = tf.shape(bond)
        gradient = tf.zeros(shape=bond_shape)
        loop_vars = [0, f_sub_one, C, C1s, C2s, bond_shape, modified_Aphis, gradient]
        _, _, _, _, _, gradient = tf.while_loop(cond = full_chain_cond, body = self._find_gradient, loop_vars=loop_vars,
                                                shape_invariants = [tf.TensorShape([]), tf.shape(C), tf.shape(f),
                                                                    tf.TensorShape(None), tf.TensorShape(None),
                                                                    tf.shape(bond_shape), tf.TensorShape(None),
                                                                    bond_shape],
                                                parallel_iterations = 10,
                                                name = "findDiscriminatorGradients")



        # perform gradient descent on the bond
        label_bond = self.rate_of_change * gradient
        label_bond = tf.clip_by_value(label_bond, -(self.cutoff), self.cutoff)
        updated_bond = tf.add(bond, label_bond)

        # calculate the cost with the updated bond
        x1 = self._get_xs(updated_bond, C)
        phi1 = tf.stack([self._phi_part, x1], axis=1)
        predictions1 = discriminator.predict(phi1)
        cost1 = discriminator.cost(predictions1, _desired_labels)
        cost = discriminator.cost(f, _desired_labels)
        if self.verbose != 0:
            cost1 = tf.Print(cost1, [cost, cost1], first_n=self.verbose, message='cost & updated cost')
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond

    def _find_gradient(self, counter, f_sub_one, C, C1s, C2s, bond_shape, modified_Aphis, gradient):
        _special_node_loc = self.delegate.discriminator._special_node_loc
        _desired_labels = self.delegate._desired_labels
        length = self.d_output
        C1 = C1s.read(counter - 1)
        C2 = C2s.read(counter + 1)
        Aphi = modified_Aphis.read(counter)
        alpha_case_0_non_spec = lambda: tf.einsum('ti,tli->tl', Aphi, C2)
        alpha_case_0_spec = lambda: tf.einsum('tli,ti->tl', Aphi, C2)
        alpha_case_l_non_spec = lambda: tf.einsum('tli,ti->tl',  tf.einsum('tij,tlj->tli', Aphi, C2), C1)
        alpha_case_mid_spec = lambda: tf.einsum('tli,ti->tl', tf.einsum('tlij,tj->tli', Aphi, C2), C1)
        alpha_case_r_non_spec = lambda: tf.einsum('ti,tli->tl',  tf.einsum('tij,tj->ti', Aphi, C2), C1)
        alpha_case_last_non_spec = lambda: tf.einsum('ti,tli->tl', Aphi, C1)
        alpha_case_last_spec = lambda: tf.einsum('tli,ti->tl', Aphi, C1)
        cond_0 = tf.equal(0, counter)
        cond_is_spec = tf.equal(counter, _special_node_loc)
        cond_not_spec = tf.logical_not(cond_is_spec)
        cond_before_spec = tf.less(counter, _special_node_loc)
        cond_after_spec = tf.greater(counter, _special_node_loc)
        cond_last = tf.equal(counter, self.generator.d_output-1)
        cond_0_non_spec = tf.logical_and(cond_0, cond_not_spec)
        cond_0_spec = tf.logical_and(cond_0, cond_is_spec)
        cond_l_non_spec = tf.logical_and(cond_before_spec, cond_not_spec)
        cond_r_non_spec = tf.logical_and(cond_after_spec, cond_not_spec)
        cond_last_non_spec = tf.logical_and(cond_last, cond_not_spec)
        cond_last_spec = tf.logical_and(cond_last, cond_is_spec)
        alpha = tf.case({cond_0_non_spec: alpha_case_0_non_spec,
                        cond_0_spec: alpha_case_0_spec,
                        cond_l_non_spec: alpha_case_l_non_spec,
                        cond_r_non_spec: alpha_case_r_non_spec,
                        cond_last_non_spec: alpha_case_last_non_spec,
                        cond_last_spec: alpha_case_last_spec},
                        default = alpha_case_mid_spec, exclusive = True)
        reduced_all_but_C = tf.einsum('tl,tl,tl->t', _desired_labels, alpha, f_sub_one)
        gradient_part = tf.tensordot(reduced_all_but_C, C, [[0], [0]])
        indices = [[counter]]
        scatter = tf.scatter_nd(indices, gradient_part, bond_shape)
        gradient = gradient + scatter
        counter += 1
        return [counter, f, C, C1s, C2s, bond_shape, modified_Aphis, gradient]

    def _chain_multiply_r(self, counter, nodes, phis, C1s, prev_C1):
        with tf.name_scope('chain_multiply_right'):
            node = nodes.read(counter)
            node.set_shape([self.delegate.discriminator.d_feature, None, None])
            contracted_node = tf.tensordot(phis[counter], node,
                                           [[1], [0]])
            C1 = tf.einsum('ti,tij->tj', prev_C1, contracted_node)
            C1s = C1s.write(counter, C1)
            counter = counter + 1
        return [counter, nodes, xs, C1s, C1]

    def _chain_multiply_l(self, counter, nodes, phis, C2s, prev_C2):
        with tf.name_scope('chain_multiply_left'):
            node = nodes.read(counter)
            node.set_shape([self.delegate.discriminator.d_feature, None, None])
            contracted_node = tf.tensordot(phis[counter], node,
                                           [[1], [0]])
            C2 = tf.einsum('tij,tj->ti', contracted_node, prev_C2)
            C2s = C1s.write(counter, C2)
            counter = counter - 1
        return [counter, nodes, xs, C1s, C1]

    def _chain_multiply_r2(self, counter, nodes, phis, C1s, prev_C1):
        # Use this function instead of chain_multiply_r if after the special node location
        with tf.name_scope('chain_multiply_right2'):
            node = nodes.read(counter)
            node.set_shape([self.delegate.discriminator.d_feature, None, None])
            contracted_node = tf.tensordot(phis[counter], node,
                                           [[1], [0]])
            C1 = tf.einsum('tli,tij->tlj', prev_C1, contracted_node)
            C1s = C1s.write(counter, C1)
            counter = counter + 1
        return [counter, nodes, xs, C1s, C1]

    def _chain_multiply_l2(self, counter, nodes, phis, C2s, prev_C2):
        # Use this function instead of chain_multiply_l if after the special node location
        with tf.name_scope('chain_multiply_left2'):
            node = nodes.read(counter)
            node.set_shape([self.delegate.discriminator.d_feature, None, None])
            contracted_node = tf.tensordot(phis[counter], node,
                                           [[1], [0]])
            C2 = tf.einsum('tij,tlj->tli', contracted_node, prev_C2)
            C2s = C1s.write(counter, C2)
            counter = counter - 1
        return [counter, nodes, xs, C1s, C1]

    def _find_Aphi(self, counter, Axs, As, phis):
        A = As.read(counter)
        x = xs[counter]
        Ax = A.tensordot(x, A, [[1], [0]])
        Axs.write(counter, Ax)
        counter = counter + 1
        return [counter, Axs, As, xs]


