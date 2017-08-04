import tensorflow as tf
import numpy as np
from mps import MPS
import utils

# TODO: replace all things referencing self.MPS with network

class TensorGANOptimizer(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_rate = tf.placeholder(tf.float32, shape=[])
        self.generator_rate = tf.placeholder(tf.float32, shape = [])
        self.tree_made = False
        self._true_features = tf.placeholder(tf.float32, shape=[self.discriminator.input_size, None, self.discriminator.d_feature])
        self._true_labels = tf.placeholder(tf.float32, shape=[None, self.discriminator.d_output])
        false_labels = None #TODO: IMPLEMENT this
        self._label = tf.stack([self._true_labels, false_labels])
        false_features = None #TODO:IMPLEMENT
        self._feature = tf.stack([self._true_labels, false_features], axis = 1)
        self.batch_size = tf.shape(self._feature)[1]

    def train_step(self):
        self.seed = tf.random_uniform([self.generator.input_size, self.batch_size, self.generator.d_feature])
        self.train_discriminator_step()
        self.train_generator_step()
        tree_made = True

    def _setup_optimization(self, network):
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
        nodes = network.nodes
        special_loc = network._special_node_loc
        batch_size = tf.shape(feature)[1]

        with tf.name_scope("setup_optimization"):


            C1 = tf.tile(network.start_node, [batch_size, 1])
            C1s = tf.TensorArray(tf.float32, size=network.input_size, infer_shape=False, clear_after_read=False)
            C1s = C1s.write(0, C1)
            cond = lambda c, *args: tf.less(c, special_loc)
            _, _, self.C1s = tf.while_loop(cond=cond, body=self._find_C1, loop_vars=[0, C1, C1s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC1")

            C2s = tf.TensorArray(tf.float32, size=network.input_size, infer_shape=False, clear_after_read=False)
            C2 = tf.tile(network.end_node, [batch_size, 1])
            C2s = C2s.write(network.input_size-1, C2)
            cond = lambda counter, *args: tf.greater(counter, special_loc+1)
            _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[network.input_size-1, C2, C2s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=10,
                                           name="initialFindC2")

    def _find_C1(self, counter, C1, C1s):
        """

        :param self:
        :param counter:
        :param C1:
        :param C1s:
        :return:
        """
        '''
        finds new C1, and write into C1s[counter]
        '''
        node = self.MPS.nodes.read(counter)
        node.set_shape([self.MPS.d_feature, None, None])
        input_leg = self._feature[counter]
        # contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        contracted_node = tf.tensordot(input_leg, node, [[1], [0]])
        C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
        C1s = C1s.write(counter+1, C1)
        counter = counter + 1
        return [counter, C1, C1s]

    def _find_C2(self, counter, prev_C2, C2s):
        """

        :param self:
        :param counter:
        :param prev_C2:
        :param C2s:
        :return:
        """
        '''
        finds new C2, and write into C2s[counter]
        '''
        node2 = self.MPS.nodes.read(counter)
        node2.set_shape([self.MPS.d_feature, None, None])
        # contracted_node2 = tf.einsum('mij,tm->tij', node2, self._feature[loc2])  # CHECK einsum
        contracted_node2 = tf.tensordot(self._feature[counter], node2, [[1], [0]])
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(counter-1, new_C2)
        counter = counter - 1
        return [counter, new_C2, C2s]

    def train(self, data_source, n_step, generator_rate, discriminator_rate,
              initial_generator_weights = None, initial_discriminator_weights = None):
        if not self.tree_made:
            self.train_step()
            test_acc, test_sample = self._test_step(self._feature, self._label, self.batch_size)
        self.generator_weights = initial_generator_weights
        self.discriminator_weights = initial_discriminator_weights
        trained_generator_weights = utils.list_from(self.generator.weights, self.generator.input_size)
        trained_discriminator_weights = utils.list_from(self.discriminator.weights, self.discriminator.input_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_step):
                (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)
                self.feed_dict = self.discriminator.create_feed_dict(self.discriminator_weights)
                self.feed_dict.update(self.generator.create_feed_dict(self.generator_weights))
                self.feed_dict[self._true_features] = batch_feature
                self.feed_dict[self._label_true] = batch_label
                self.feed_dict[self.discriminator_rate] = discriminator_rate
                self.feed_dict[self.generator_rate] = generator_rate
                to_eval = [trained_discriminator_weights, trained_generator_weights, test_acc, test_sample]
                self.discriminator_weights, self.generator_weights, accuracy, sample = sess.run(to_eval,
                                                                              feed_dict = self.feed_dict)
                print('Discriminator accuracy:', accuracy)
                with open('generatorWeights', 'wb') as fp:
                    pickle.dump(self.generator_weights, fp)
                with open('discriminatorWeights', 'wb') as fp:
                    pickle.dump(self.discriminator_weights, fp)

    def _test_step(self, feature, label, batch_size):
        seed = tf.random_uniform([self.generator.input_size, batch_size, self.generator.d_feature])
        false_features, false_labels = self.generator.create(seed = seed)
        predictions_true = self.discriminator.predict(feature)
        predictions_false = self.discriminator.predict(false_features)
        accuracy_true = self.discriminator.accuracy(predictions_true, label)
        accuracy_false = self.discriminator.accuracy(predictions_false, false_labels)
        accuracy = (accuracy_true + accuracy_false)*0.5
        sample = tf.transpose(false_features, perm = [1, 0, 2])[0]
        return accuracy, sample

    def train_step_discriminator(self):
    """
    A single step of training. Interface with this if you need to chain training with other tensorflow operations.
    If not, it is recommended to just use the train function.

    :return: the accuracy as calculated at the end of a training step.
    """

        with tf.name_scope("train_step"):
            # Create updated_nodes and fill in the first half from current one
            updated_nodes = self._duplicate_nodes(self.discriminator.nodes, 0, self.discriminator._special_node_loc)
            original_special_node_loc = self.discriminator._special_node_loc

            # First half-sweep
            updated_nodes = self._sweep_right(self.discriminator._special_node_loc, self.discriminator.input_size-1)
            self.discriminator.nodes = updated_nodes
            self.discriminator._special_node_loc = self.discriminator.input_size - 1

            # First back-sweep
            updated_nodes = self._duplicate_nodes(self.discriminator.nodes, 0, 0)
            updated_nodes = self._sweep_left()
            self.discriminator.nodes = updated_nodes
            self.discriminator._special_node_loc = 0

            # Second half-sweep
            updated_nodes = self._duplicate_nodes(self.discriminator.nodes, original_special_node_loc + 1,
                                                       self.discriminator.input_size)
            C1 = self.C1s.read(0)
            self.C1s = tf.TensorArray(tf.float32, size=self.discriminator.input_size, dynamic_size=False, infer_shape=False,
                                      clear_after_read=False)
            self.C1s = self.C1s.write(0, C1)
            updated_nodes = self._sweep_right(0, original_special_node_loc)
            self.discriminator.nodes = updated_nodes
            self.discriminator._special_node_loc = original_special_node_loc

    def train_step_generator(self):
    """
    A single step of training. Interface with this if you need to chain training with other tensorflow operations.
    If not, it is recommended to just use the train function.

    :return: the accuracy as calculated at the end of a training step.
    """

        with tf.name_scope("train_step"):
            # Create updated_nodes and fill in the first half from current one
            updated_nodes = self._duplicate_nodes(self.generator.nodes, 0, self.generator._special_node_loc)
            original_special_node_loc = self.generator._special_node_loc

            # First half-sweep
            updated_nodes = self._sweep_right(self.generator._special_node_loc, self.generator.input_size-1)
            self.generator.nodes = updated_nodes
            self.generator._special_node_loc = self.generator.input_size - 1

            # First back-sweep
            updated_nodes = self._duplicate_nodes(self.generator.nodes, 0, 0)
            updated_nodes = self._sweep_left()
            self.generator.nodes = updated_nodes
            self.generator._special_node_loc = 0

            # Second half-sweep
            updated_nodes = self._duplicate_nodes(self.generator.nodes, original_special_node_loc + 1,
                                                       self.generator.input_size)
            C1 = self.C1s.read(0)
            self.C1s = tf.TensorArray(tf.float32, size=self.generator.input_size, dynamic_size=False, infer_shape=False,
                                      clear_after_read=False)
            self.C1s = self.C1s.write(0, C1)
            updated_nodes = self._sweep_right(0, original_special_node_loc)
            self.generator.nodes = updated_nodes
            self.generator._special_node_loc = original_special_node_loc

    def _sweep_left(self, network):
        """

        :param self:
        :return:
        """
        # read second from end node
        n1 = network.nodes.read(network._special_node_loc)
        n1.set_shape([network.d_output, network.d_feature, None, None])

        C2 = self.C2s.read(network.input_size - 1)
        self.C2s = tf.TensorArray(tf.float32, size=network.input_size, infer_shape=False, clear_after_read=False)
        self.C2s = self.C2s.write(network.input_size - 1, C2)
        cond = lambda counter, *args: tf.greater(counter, 0)
        wrapped = [network.input_size-1, self.C2s, network.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None])]

        _, self.C2s, updated_nodes, n1 = tf.while_loop(cond=cond, body=self._update_left,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10,
                                                            name="leftSweep")
        .updated_nodes = updated_nodes.write(0, n1)
        return updated_nodes

    def _sweep_right(self, network, from_index, to_index):
        """

        :param self:
        :param from_index:
        :param to_index:
        :return:
        """
        n1 = network.nodes.read(from_index)
        n1.set_shape([network.d_output, network.d_feature, None, None])

        cond = lambda counter, *args: tf.less(counter, to_index)
        wrapped = [from_index, self.C1s, network.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape([None, None, None, None])]
        _, self.C1s, network.nodes, n1 = tf.while_loop(cond=cond, body=self._update_right,
                                                            loop_vars=wrapped,
                                                            shape_invariants=shape_invariants,
                                                            parallel_iterations=10, name="rightSweep")
        network.nodes = network.nodes.write(to_index, n1)
        return updated_nodes

    def _update_left_gen(self, network, feature = self._feature):
        _network = network
        feature = feature
        def _update_left(counter, C2s, updated_nodes, previous_node):
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
                n2 = _network.nodes.read(counter - 1)
                n2.set_shape([_network.d_feature, None, None])

                # Calculate the C matrix
                C2 = C2s.read(counter)
                C1 = self.C1s.read(counter - 1)
                C1.set_shape([None, None])
                C2.set_shape([None, None])
                input1 = _feature[counter - 1]
                input2 = _feature[counter]

                # Calculate the bond
                bond = tf.einsum('nkj,lmji->lmnik', n2, n1)

                C = self._calculate_C(C2, C1, input2, input1, network)

                # update the bond
                updated_bond = self._update_bond(bond, C)

                # Decompose the bond
                aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
                aj = tf.transpose(aj, perm=[0, 2, 1])
                aj1 = tf.transpose(aj1, perm=[1, 2, 3, 0])

                # Transpose the values and add to the new variables
                updated_nodes = updated_nodes.write(counter, aj)
                with tf.name_scope("tensordotcontracted_aj"):
                    # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                    contracted_aj = tf.tensordot(_feature[counter], aj, [[1], [0]])
                with tf.name_scope("einsumC2"):
                    C2 = tf.einsum('tij,tj->ti', contracted_aj, C2)
                C2s = C2s.write(counter - 1, C2)
                #counter = tf.Print(counter, [counter])
                updated_counter = counter - 1

            return [updated_counter, C2s, updated_nodes, aj1]
        return _update_left

    def _update_right_gen(self, network, feature = self._feature):
        _network = network
        _feature = feature
        def _update_right(self, counter, C1s, updated_nodes, previous_node):
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
                n2 = network.nodes.read(counter + 1)
                n2.set_shape([network.d_feature, None, None])

                # Calculate the C matrix
                C2 = self.C2s.read(counter+1)
                C1 = C1s.read(counter)
                C1.set_shape([None, None])
                C2.set_shape([None, None])
                input1 = _feature[counter]
                input2 = _feature[counter + 1]

                # Calculate the bond
                bond = tf.einsum('lmij,njk->lmnik', n1, n2)
                # bond = tf.transpose(tf.tensordot(n1, n2, [[3],[1]]), [0, 1, 3, 2, 4])
                # einsum is actually faster in this case

                C = self._calculate_C(C1, C2, input1, input2, network)

                # Update the bond
                updated_bond = self._update_bond(bond, C)

                # Decompose the bond
                aj, aj1 = self._bond_decomposition(updated_bond, self.max_size)
                aj1 = tf.transpose(aj1, perm=[1, 2, 0, 3])

                # Transpose the values and add to the new variables
                updated_nodes = updated_nodes.write(counter, aj)

                with tf.name_scope("tensordotcontracted_aj"):
                    # contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
                    contracted_aj = tf.tensordot(_feature[counter], aj, [[1], [0]])
                with tf.name_scope("einsumC1"):
                    C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
                C1s = C1s.write(counter+1, C1)
                #counter = tf.Print(counter, [counter])
                updated_counter = counter + 1

            return [updated_counter, C1s, updated_nodes, aj1]

    def _calculate_C(self, C1, C2, input1, input2, network):
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
            input1 = tf.reshape(input1, [self.batch_size, network.d_feature, 1, 1, 1])
            input2 = tf.reshape(input2, [self.batch_size, 1, network.d_feature, 1, 1])
            intermediate_1 = C1 * C2
            intermediate_2 = input1 * input2
            C = intermediate_1 * intermediate_2

        return C

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
            h = tf.nn.softmax(f)
        with tf.name_scope("reduce_sumcost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._label, logits=f))
            # 0.5 * tf.reduce_sum(tf.square(f-self._label))

        return h, cost

    def _update_discriminator_bond(self, bond, C):
        # obtain the original cost
        f, cost = self._get_f_and_cost(bond, C)

        # perform gradient descent on the bond
        with tf.name_scope("tensordotgradient"):
            # gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)
            gradient = tf.tensordot(self._label - f, C, [[0], [0]]) + 2 * self.reg * bond
        label_bond = self.rate_of_change * gradient
        label_bond = tf.clip_by_value(label_bond, -(self.cutoff), self.cutoff)
        updated_bond = tf.add(bond, label_bond)

        # calculate the cost with the updated bond
        f1, cost1 = self._get_f_and_cost(updated_bond, C)
        if self.verbose != 0:
            cost1 = tf.Print(cost1, [cost, cost1], first_n=self.verbose, message='cost & updated cost')
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond

    def _update_generator_bond(self, bond, C):
        # obtain the original cost
        g = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
        x = tf.nn.softmax(g)
        x_prime = tf.diag(x)
        f = self.discriminator.predict(x)
        f_prime = self.discriminator.predict()

        # perform gradient descent on the bond
        with tf.name_scope("tensordotgradient"):
            # gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)

            gradient = tf.tensordot(self._label - f, C, [[0], [0]]) + 2 * self.reg * bond
        label_bond = self.rate_of_change * gradient
        label_bond = tf.clip_by_value(label_bond, -(self.cutoff), self.cutoff)
        updated_bond = tf.add(bond, label_bond)

        # calculate the cost with the updated bond
        f1, cost1 = self._get_f_and_cost(updated_bond, C)
        if self.verbose != 0:
            cost1 = tf.Print(cost1, [cost, cost1], first_n=self.verbose, message='cost & updated cost')
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond),
                               false_fn=(lambda: tf.Print(bond, [cost, cost1], message='Gradient may be too big/too small')))

        return updated_bond

    def _update_generator_bond_helper(self, bond, C):
        g = tf.tensordot(C, bond, [[1, 2, 3, 4], [1, 2, 3, 4]])
        x = tf.sigmoid(g)
        f = self.discriminator.predict(x)
        f_prime = self.discriminator.predict()


    def _make_new_nodes(self, nodes):
        """

        :param self:
        :param nodes:
        :return:
        """
        size = nodes.size()
        new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)
        new_nodes = new_nodes.write(0, nodes.read(size - 1))
        new_nodes = new_nodes.write(size - 1, nodes.read(0))
        return new_nodes

    def _duplicate_nodes(self, nodes, from_index, to_index):
        """

        :param self:
        :param nodes:
        :param from_index:
        :param to_index:
        :return:
        """
        '''
        duplicate the nodes in the range (from_index, to_index)
        '''
        with tf.name_scope("duplicatingNodes"):
            size = nodes.size()
            new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)

            from_index = tf.cond(tf.less(from_index, 0), lambda: 0, lambda: from_index)
            to_index = tf.cond(tf.greater(to_index, size), lambda: size, lambda: to_index)
            cond = lambda index, a, b: tf.less(index, to_index)

            _, _, new_nodes = tf.while_loop(cond=cond, body=self._transfer_to_array,
                                            loop_vars=[from_index, nodes, new_nodes], name="duplicate_loop")
        return new_nodes

    def _transfer_to_array(self, index, old_nodes, new_nodes):
        """

        :param self:
        :param index:
        :param old_nodes:
        :param new_nodes:
        :return:
        """
        old_node = old_nodes.read(index)
        new_nodes = new_nodes.write(index, old_node)
        index += 1
        return (index, old_nodes, new_nodes)

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
            if self.verbose != 0:
                m = tf.Print(m, [m, s[m-5:m]], first_n=self.verbose, summarize=5, message='bond: ')

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
