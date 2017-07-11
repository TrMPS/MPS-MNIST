import tensorflow as tf
import numpy as np
import preprocessing
from mps import MPS
import pickle
import utils


def list_from(tensorArray, length):
    arr = tensorArray
    result_list = []
    with tf.name_scope("createlist"):
        for i in range(length):
            result_list.append(arr.read(i))
    return result_list


class MPSOptimizer(object):

    def __init__(self, MPSNetwork, bond_dim, grad_func, rate_of_change=1000, cutoff=10 ** (-5)):
        self.MPS = MPSNetwork
        self.rate_of_change = rate_of_change
        self.bond_dim = bond_dim
        self.grad_func = grad_func
        self.cutoff = cutoff
        self._feature = tf.placeholder(tf.float32, shape=[input_size, None, self.MPS.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self.MPS._setup_nodes(self._feature)
        self._setup_optimization()
        _ = self.train_step()

    def train(self, data_source, batch_size, n_step, log_to_tensorboard=None, initial_weights=None):
        _log_to_tensorboard = log_to_tensorboard
        if log_to_tensorboard is None:
            _log_to_tensorboard = False

        run_options = []
        run_metadata = []
        if log_to_tensorboard:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        self.feed_dict = None
        self.test = None
        f = self.MPS.predict(self._feature)
        cost = self.MPS.cost(f, self._label)
        accuracy = self.MPS.accuracy(f, self._label)
        test_result = list_from(self.updated_nodes, length=self.MPS.input_size)
        self.test = initial_weights

        with tf.Session() as sess:
            if _log_to_tensorboard:
                writer = tf.summary.FileWriter("output", sess.graph)
            for i in range(n_step):
                sess.run(tf.global_variables_initializer())

                (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)
                self.feed_dict = self.MPS.create_feed_dict(self.test)
                self.feed_dict[self._feature] = batch_feature
                self.feed_dict[self._label] = batch_label
                train_cost, prediction, self.test, train_accuracy = sess.run([cost, f, test_result, accuracy],
                                                                             feed_dict=self.feed_dict)
                                                                             #options=run_options,
                                                                             #run_metadata=run_metadata)
                print('step {}, training cost {}, accuracy {}'.format(i, train_cost, train_accuracy))
                print("prediction:" + str(prediction[0]))

                print(self.test[0])
                self.feed_dict = {self._feature: batch_feature, self._label: batch_label}
                for index, element in enumerate(self.test):
                    self.feed_dict[self.MPS.nodes_list[index]] = element
                if _log_to_tensorboard:
                    writer.add_run_metadata(run_metadata, 'step' + str(i))
                with open('weights', 'wb') as fp:
                    pickle.dump(self.test, fp)
            if _log_to_tensorboard:
                writer.close()

    def _setup_optimization(self):
        '''
        C1s: size = input_size - 2 (as the last one is kept redundant) 
        C2s: size = input_size - 2 (first one redundant)

        writes C1 from 0 to special_loc-1 
        writes C2 from special_loc to size-3 (corresponds to special_loc+2 to size-1 the nodes)
        '''
        feature = self._feature
        nodes = self.MPS.nodes
        special_loc = self.MPS._special_node_loc
        with tf.name_scope("setup_optimization"):
            n1 = nodes.read(0)
            n1.set_shape([None, None])
            C1 = tf.einsum('ni,tn->ti', n1, feature[0])
            C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size - 2, infer_shape=False, clear_after_read=False)
            C1s = C1s.write(0, C1)
            cond = lambda c, *args: tf.less(c, special_loc)
            _, _, self.C1s = tf.while_loop(cond=cond, body=self._find_C1, loop_vars=[1, C1, C1s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)],
                                           parallel_iterations=1,
                                           name="initialFindC1")

            nlast = nodes.read(nodes.size() - 1)
            nlast.set_shape([None, None])
            C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size - 2, infer_shape=False, clear_after_read=False)
            C2 = tf.einsum('mi,tm->ti', nlast, feature[-1])
            C2s = C2s.write(self.MPS.input_size - 3, C2)
            cond = lambda counter, *args: tf.less(counter, self.MPS.input_size - special_loc - 2)
            _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[1, C2, C2s],
                                           shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                             tf.TensorShape(None)], parallel_iterations=1,
                                           name="initialFindC2")

    def _find_C1(self, counter, C1, C1s):
        '''
        finds new C1, and write into C1s[counter]
        '''
        node = self.MPS.nodes.read(counter)
        node.set_shape([self.MPS.d_feature, None, None])
        input_leg = self._feature[counter]
        contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
        C1s = C1s.write(counter, C1)
        counter = counter + 1
        return [counter, C1, C1s]

    def _find_C2(self, counter, prev_C2, C2s):
        '''
        finds new C2, and write into C2s[counter]
        '''
        loc2 = self.MPS.input_size - 1 - counter
        node2 = self.MPS.nodes.read(loc2)
        node2.set_shape([self.MPS.d_feature, None, None])
        contracted_node2 = tf.einsum('mij,tm->tij', node2, self._feature[loc2])  # CHECK einsum
        updated_counter = counter + 1
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(self.MPS.input_size - 3 - counter, new_C2)
        return [updated_counter, new_C2, C2s]

    def train_step(self):
        with tf.name_scope("train_step"):
            # Create updated_nodes and fill in the first half from current one
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, self.MPS._special_node_loc)
            original_special_node_loc = self.MPS._special_node_loc

            # First half-sweep
            self.updated_nodes = self._sweep_right(self.MPS._special_node_loc, self.MPS.input_size - 2)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = self.MPS.nodes.size() - 2

            # First back-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, 0)
            self.updated_nodes = self._sweep_left()
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = 1
            
            # Second half-sweep
            self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, original_special_node_loc + 1,
                                                       self.MPS.nodes.size() + 10)
            C1 = self.C1s.read(0)
            self.C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size - 2, dynamic_size=True, infer_shape=False,
                                      clear_after_read=False)
            self.C1s = self.C1s.write(0, C1)
            self.updated_nodes = self._sweep_right(1, original_special_node_loc)
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = original_special_node_loc
            
            # accuracy
            f = self.MPS.predict(self._feature)
            accuracy = self.MPS.accuracy(f, self._label)

        return accuracy

    def _sweep_left(self):
        # read second from end node
        n1 = self.MPS.nodes.read(self.MPS._special_node_loc)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])

        C2 = self.C2s.read(self.MPS.input_size - 3)
        self.C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size - 2, infer_shape=False, clear_after_read=False)
        self.C2s = self.C2s.write(self.MPS.input_size - 3, C2)
        cond = lambda counter, b, c, d, e, f: tf.greater(counter, 1)
        wrapped = [self.MPS.nodes.size() - 2, self.C1s, self.C2s, self.updated_nodes, self.MPS.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([None, None, None, None])]

        counter, self.C1s, self.C2s, self.updated_nodes, _, n1 = tf.while_loop(cond=cond, body=self._update_left,
                                                                               loop_vars=wrapped,
                                                                               shape_invariants=shape_invariants,
                                                                               parallel_iterations=1,
                                                                               name="backwardsSweep")
        with tf.control_dependencies([counter]):
            self.updated_nodes = self.updated_nodes.write(1, n1)
        return self.updated_nodes

    def _sweep_right(self, from_index, to_index):
        n1 = self.MPS.nodes.read(from_index)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
        cond = lambda counter, b, c, d, e, f: tf.less(counter, to_index)
        wrapped = [from_index, self.C1s, self.C2s, self.updated_nodes, self.MPS.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([None, None, None, None])]

        _, self.C1s, self.C2s, self.updated_nodes, _, n1 = tf.while_loop(cond=cond, body=self._update_right,
                                                                         loop_vars=wrapped,
                                                                         shape_invariants=shape_invariants,
                                                                         parallel_iterations=1, name="rightSweep")
        self.updated_nodes = self.updated_nodes.write(to_index, n1)
        return self.updated_nodes

    def _update_left(self, counter, C1s, C2s, updated_nodes, nodes, previous_node):

        with tf.name_scope("update_left"):
            # Read in the nodes 
            n1 = previous_node
            n2 = nodes.read(counter - 1)
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the bond 
            bond = tf.einsum('nkj,lmji->lmnik', n2, n1)

            # Calculate the C matrix 
            C2 = C2s.read(counter - 1)
            C2.set_shape([None, None])
            C1 = C1s.read(counter - 2)
            C1.set_shape([None, None])
            with tf.name_scope("einsumC"):
                C = tf.einsum('ti,tk,tm,tn->tmnik', C2, C1, self._feature[counter], self._feature[counter - 1])

            # update the bond 
            updated_bond = self._update_bond(bond, C)

            # Decompose the bond 
            aj, aj1 = self._bond_decomposition(updated_bond, self.bond_dim)
            aj = tf.transpose(aj, perm=[0, 2, 1])
            aj1 = tf.transpose(aj1, perm=[0, 1, 3, 2])

            # Transpose the values and add to the new variables 
            updated_nodes = updated_nodes.write(counter, aj)
            with tf.name_scope("einsumcontracted_aj"):
                contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
            with tf.name_scope("einsumC2"):
                C2 = tf.einsum('tij,tj->ti', contracted_aj, C2)
            C2s = C2s.write(counter - 2, C2)
            updated_counter = counter - 1

        return [updated_counter, C1s, C2s, updated_nodes, nodes, aj1]

    def _update_right(self, counter, C1s, C2s, updated_nodes, nodes, previous_node):
        with tf.name_scope("update_right"):
            # Read in the nodes 
            n1 = previous_node
            n2 = nodes.read(counter + 1)
            n2.set_shape([self.MPS.d_feature, None, None])
    
            # Calculate the bond 
            bond = tf.einsum('lmij,njk->lmnik', n1, n2)
    
            # Calculate the C matrix 
            C2 = C2s.read(counter)
            C1 = C1s.read(counter - 1)
            C2.set_shape([None, None])
            C1.set_shape([None, None])
            with tf.name_scope("einsumC"):
                C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, self._feature[counter], self._feature[counter + 1])
    
            # Update the bond 
            updated_bond = self._update_bond(bond, C)
    
            # Decompose the bond 
            aj, aj1 = self._bond_decomposition(updated_bond, self.bond_dim)
    
            # Transpose the values and add to the new variables 
            updated_nodes = updated_nodes.write(counter, aj)
    
            with tf.name_scope("einsumcontracted_aj"):
                contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
            with tf.name_scope("einsumC1"):
                C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
            C1s = C1s.write(counter, C1)
            updated_counter = counter + 1
    
        return [updated_counter, C1s, C2s, updated_nodes, nodes, aj1]

    def _get_f_and_cost(self, bond, C):
        with tf.name_scope("einsumf"):
            f = tf.einsum('lmnik,tmnik->tl', bond, C)
        with tf.name_scope("einsumcost"):
            cost = 0.5 * tf.einsum('tl,tl->', f-self._label, f-self._label)

        return f, cost

    def _update_bond(self, bond, C):
        # obtain the original cost
        f, cost = self._get_f_and_cost(bond, C)

        # perform gradient descent on the bond 
        with tf.name_scope("einsumgradient"):
            gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)
        label_bond = self.rate_of_change * gradient
        label_bond = tf.clip_by_value(label_bond, -(self.cutoff), self.cutoff)
        updated_bond = tf.add(bond, label_bond)
        
        # calculate the cost with the updated bond
        f1, cost1 = self._get_f_and_cost(updated_bond, C)
        cost1 = tf.Print(cost1, [cost, cost1], message='cost and updated cost')
        cond_change_bond = tf.less(cost1, cost)
        updated_bond = tf.cond(cond_change_bond, true_fn=(lambda: updated_bond), false_fn=(lambda: bond))

        return updated_bond

    def _make_new_nodes(self, nodes):
        size = nodes.size()
        new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)
        new_nodes = new_nodes.write(0, nodes.read(size - 1))
        new_nodes = new_nodes.write(size - 1, nodes.read(0))
        return new_nodes

    def _duplicate_nodes(self, nodes, from_index, to_index):
        '''
        duplicate the nodes in the range (from_index, to_index)
        '''
        with tf.name_scope("duplicatingNodes"):
            size = nodes.size()
            new_nodes = tf.TensorArray(tf.float32, size=size, infer_shape=False, clear_after_read=False)

            from_index = tf.cond(tf.equal(from_index, 0), lambda: 1, lambda: from_index)
            to_index = tf.cond(tf.greater(to_index, size - 1), lambda: size - 1, lambda: to_index)
            cond = lambda index, a, b: tf.less(index, to_index)

            _, _, new_nodes = tf.while_loop(cond=cond, body=self._transfer_to_array,
                                            loop_vars=[from_index, nodes, new_nodes], name="duplicate_loop")
            new_nodes = new_nodes.write(0, nodes.read(0))
            new_nodes = new_nodes.write(size - 1, nodes.read(size - 1))
        return new_nodes

    def _transfer_to_array(self, index, old_nodes, new_nodes):
        old_node = old_nodes.read(index)
        new_nodes = new_nodes.write(index, old_node)
        index += 1
        return (index, old_nodes, new_nodes)

    def _bond_decomposition(self, bond, max_size, min_size=None, threshold=None):
        """
        Decomposes bond, so that the next step can be done.
        :param bond:
        :param m:
        :return:
        """
        if threshold is None:
            _threshold = 10 ** (-8)
        else:
            _threshold = threshold
        if min_size is None:
            min_size = 3
        else:
            min_size = min_size
        with tf.name_scope("bond_decomposition"):
            bond_reshaped = tf.transpose(bond, perm=[1, 3, 0, 2, 4])

            dims = tf.shape(bond_reshaped)
            l_dim = dims[0] * dims[1]
            r_dim = dims[2] * dims[3] * dims[4]
            bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
            s, u, v = tf.svd(bond_flattened)
            filtered_u = utils.check_nan(u, 'u', replace_nan=True)
            filtered_v = utils.check_nan(v, 'v', replace_nan=True)

            filtered_s = tf.boolean_mask(s, tf.greater(s, _threshold))
            s_size = tf.size(filtered_s)

            case1 = lambda: min_size
            case2 = lambda: max_size
            case3 = lambda: s_size
            m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3,
                        exclusive=True)

            # make s into a matrix
            s_mat = tf.diag(s[0:m])

            # make u, v into suitable matrices
            u_cropped = filtered_u[:, 0:m]
            v_cropped = tf.transpose(filtered_v[:, 0:m])

            # make a_ 
            a_prime_j = tf.reshape(u_cropped, [dims[0], dims[1], m])
            sv = tf.matmul(s_mat, v_cropped)
            a_prime_j1_mixed = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
            a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])

        return (a_prime_j, a_prime_j1)




if __name__ == '__main__':
    # Model parameters
    input_size = 196
    d_feature = 2
    d_output = 10
    batch_size = 1000

    bond_dim = 4
    max_size = 15
    rate_of_change = 1000
    log_to_tensorboard = False

    cutoff = 10
    n_step = 10

    data_source = preprocessing.MNISTData()

    # Initialise the model
    # with open('weights', 'rb') as fp:
    #     weights = pickle.load(fp)
    #     if len(weights) != input_size:
    #         weights = None
    weights = None
    network = MPS(bond_dim, d_feature, d_output, input_size)
    optimizer = MPSOptimizer(network, max_size, None, rate_of_change=rate_of_change, cutoff=cutoff)
    optimizer.train(data_source, batch_size, n_step, log_to_tensorboard=log_to_tensorboard, initial_weights=weights)
