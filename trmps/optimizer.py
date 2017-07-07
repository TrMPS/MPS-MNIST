import tensorflow as tf 
import numpy as np
import preprocessing
from mps import MPS
import pickle

def list_from(tensorArray, length):
    arr = tensorArray
    result_list = []
    with tf.name_scope("createlist"):
        for i in range(length):
            result_list.append(arr.read(i))
    return result_list

class MPSOptimizer(object):

    def __init__(self, MPSNetwork, bond_dim, grad_func, rate_of_change = None):
        self.MPS = MPSNetwork
        self.rate_of_change = 10**(-9)
        self.bond_dim = bond_dim
        self.grad_func = grad_func
        self._feature = tf.placeholder(tf.float32, shape=[input_size, None, self.MPS.d_feature])
        self._label = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self._setup_optimization()
        _ = self.train_step()

    def train(self, data_source, batch_size):
            if self.rate_of_change == None:
                self.rate_of_change = 0.1/(batch_size)
            self.feed_dict = None
            self.test = None
            for i in range(1):
                if self.test is not None:
                    self.MPS.load_nodes(self.test)
                    with open('weights', 'wb') as fp:
                        pickle.dump(self.test, fp)
                    print(self.test[-5])
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
    
                    (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)
                    if self.feed_dict is None:
                        self.feed_dict={self._feature: batch_feature, self._label: batch_label}
                    self.feed_dict[self._feature] = batch_feature
                    self.feed_dict[self._label] = batch_label
                    f = self.MPS.predict(self._feature)
                    cost = self.MPS.cost(f, self._label)
                    train_accuracy = cost.eval(feed_dict=self.feed_dict)
                    print('step {}, training accuracy {}'.format(i, train_accuracy))
    
                    test_result = list_from(self.updated_nodes, length = self.MPS.input_size)
                    self.test = sess.run(test_result, feed_dict=self.feed_dict)
                    #print(self.test)
                    self.feed_dict = {self._feature: batch_feature, self._label: batch_label}
                    for index, element in enumerate(self.test):
                        self.feed_dict[self.MPS.nodes_list[index]] = element
                    #train_accuracy = accuracy.eval(feed_dict=self.feed_dict)
                    #print('step {}, training accuracy {}'.format(i, train_accuracy))



    def _setup_optimization(self):
        feature = self._feature
        nodes = self.MPS.nodes
        special_loc = self.MPS._special_node_loc

        n1 = nodes.read(0)
        n1.set_shape([None,None])
        nlast = nodes.read(nodes.size() - 1)
        nlast.set_shape([None,None])

        C1 = tf.einsum('ni,tn->ti', n1, feature[0])
        C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size-2, dynamic_size = True, infer_shape=False, clear_after_read = False)
        C1s = C1s.write(0, C1)
        cond = lambda c, *args: tf.less(c, special_loc)
        _, self.C1, self.C1s = tf.while_loop(cond=cond, body=self._find_C1, loop_vars=[1, C1, C1s], 
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None,None]), tf.TensorShape(None)],
                                        parallel_iterations = 1)

        C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size-3, infer_shape=False, clear_after_read = False)
        self.C1 = tf.einsum('ni,tn->ti', n1, feature[0])
        C2 = tf.einsum('mi,tm->ti', nlast, feature[-1])
        C2s = C2s.write(self.MPS.input_size-4, C2)
        cond = lambda counter, *args: tf.less(counter, input_size - special_loc - 3)
        _, self.C2, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[0, C2, C2s], 
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]), 
                                        tf.TensorShape(None)])


    def _find_C1(self, counter, C1, C1s):
        node = self.MPS.nodes.read(counter)
        node.set_shape([self.MPS.d_feature, None, None])
        input_leg = self._feature[counter]
        contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        C1 = tf.einsum('ti,tij->tj', C1, contracted_node)
        C1s = C1s.write(counter, C1)
        counter = counter + 1 
        return [counter, C1, C1s]

    def train_step(self):
        # Create updated_nodes and fill in the first half from current one
        self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, self.MPS._special_node_loc)
        original_special_node_loc = self.MPS._special_node_loc
        # First half-sweep
        a = tf.Print(self.MPS.nodes.read(1), [self.MPS.nodes.read(1)], message = "optimizerStartedFirstSweep")
        self.updated_nodes = self._sweep_right(self.MPS._special_node_loc, self.MPS.input_size - 2)
        a = tf.Print(self.updated_nodes.read(1), [self.updated_nodes.read(1)], message = "optimizerEndedFirstSweep")
        with tf.control_dependencies([a]):
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = self.MPS.nodes.size() - 2

        # First back-sweep
        self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, 0, 0)
        self.updated_nodes = self._backwards_sweep()
        a = tf.Print(self.updated_nodes.read(1), [self.updated_nodes.read(1)], message = "optimizerEndedBackSweep")
        with tf.control_dependencies([a]):
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = 1
#
        # Second half-sweep
        self.updated_nodes = self._duplicate_nodes(self.MPS.nodes, original_special_node_loc + 1, self.MPS.nodes.size() + 10)
        C1 = self.C1s.read(0)
        self.C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size-2, dynamic_size = True, infer_shape=False, clear_after_read = False)
        self.C1s = self.C1s.write(0, C1)
        self.updated_nodes = self._sweep_right(1, original_special_node_loc)
        a = tf.Print(self.updated_nodes.read(1), [self.updated_nodes.read(1)], message = "optimizerEndedFinalHalfSweep")
        with tf.control_dependencies([a]):
            self.MPS.nodes = self.updated_nodes
            self.MPS._special_node_loc = original_special_node_loc
##
        # accuracy
        f = self.MPS.predict(self._feature)
        accuracy = self.MPS.accuracy(f, self._label)

        return accuracy

    def _backwards_sweep(self):
        # read second from end node
        n1 = self.MPS.nodes.read(self.MPS.nodes.size() - 2)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
        C2 = self.C2s.read(self.MPS.input_size-4)
        self.C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size-2, dynamic_size = True, infer_shape=False, clear_after_read = False)
        self.C2s = self.C2s.write(self.MPS.input_size-3, C2)
        cond = lambda counter, b, c, d, e, f: tf.greater(counter, 1)
        wrapped = [self.MPS.nodes.size() - 2, self.C1s, self.C2s, self.updated_nodes, self.MPS.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([None, None, None, None])]

        counter, self.C1s, self.C2s, self.updated_nodes, _, n1 = tf.while_loop(cond = cond, body = self._update_left, loop_vars = wrapped,
                                                                        shape_invariants = shape_invariants,
                                                                         parallel_iterations = 1)
        counter = tf.Print(counter, [counter], message = "counter at end of backsweep")
        with tf.control_dependencies([counter]):
            self.updated_nodes = self.updated_nodes.write(1, n1)
        return self.updated_nodes

    def _update_left(self, counter, C1s, C2s, updated_nodes, nodes, previous_node):
        # Read in the nodes 
        n1 = previous_node
        #counter = tf.Print(counter, [counter], message = "Counter")
        n2 = nodes.read(counter-1)
        n2.set_shape([self.MPS.d_feature, None, None])

        # Calculate the bond 
        #n1 = tf.Print(n1, [tf.shape(n1), tf.shape(n2)], "n1 and n2")
        bond = tf.einsum('lmji,nkj->lnmki', n1, n2)

        # Calculate the C matrix 
        C2 = C2s.read(counter - 1)
        C2.set_shape([None,None])
        C1 = C1s.read(counter - 1)
        C1.set_shape([None,None])
        C = tf.einsum('ti,tk,tm,tn->tmnik', C2, C1, self._feature[counter-1], self._feature[counter])

        # Update the bond 
        f = tf.einsum('lmnik,tmnik->tl', bond, C)
        #f = tf.Print(f, [counter, f], message = 'counter, f')
        gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)
        label_bond = self.rate_of_change * gradient
        cutoff = 10**(-5)
        label_bond = tf.clip_by_value(label_bond, -(cutoff), cutoff)
        #label_bond = tf.Print(label_bond, [label_bond], message = "label_bond")
        random_part = tf.random_uniform(tf.shape(label_bond), minval=-(cutoff/10), maxval = cutoff/10)
        label_bond = tf.add(label_bond, random_part)
        updated_bond = tf.add(bond, label_bond)

        # Decompose the bond 
        aj, aj1 = self._bond_decomposition(updated_bond, self.bond_dim)

        # Transpose the values and add to the new variables 
        updated_nodes = updated_nodes.write(counter, aj)
        contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
        C2 = tf.einsum('tij,ti->tj', contracted_aj, C2)
        C2s = C2s.write(counter -2, C2)
        updated_counter = counter-1 

        return [updated_counter, C1s, C2s, updated_nodes, nodes, aj1]



    def _sweep_right(self, from_index, to_index):
        n1 = self.MPS.nodes.read(from_index)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
        cond = lambda counter, b, c, d, e, f: tf.less(counter, to_index)
        wrapped = [from_index, self.C1s, self.C2s, self.updated_nodes, self.MPS.nodes, n1]
        shape_invariants = [tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape(None),
                            tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape([None, None, None, None])]

        _, self.C1s, self.C2s, self.updated_nodes, _, n1 = tf.while_loop(cond = cond, body = self._update_right, loop_vars = wrapped,
                                                                        shape_invariants = shape_invariants,
                                                                         parallel_iterations=1)
        self.updated_nodes = self.updated_nodes.write(to_index, n1)
        return self.updated_nodes

    def _update_right(self, counter, C1s, C2s, updated_nodes, nodes, previous_node):

        # Read in the nodes 
        #counter = tf.Print(counter, [counter], "Counter")
        n1 = previous_node
        n2 = nodes.read(counter+1)
        n2.set_shape([self.MPS.d_feature, None, None])
        #n1 = tf.Print(n1, [tf.shape(n1), tf.shape(n2)], "n1 and n2")
        # Calculate the bond 
        bond = tf.einsum('lmij,njk->lmnik', n1, n2)

        # Calculate the C matrix 
        C2 = C2s.read(counter - 1)
        C1 = C1s.read(counter - 1)
        C2.set_shape([None, None])
        C1.set_shape([None, None])
        C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, self._feature[counter], self._feature[counter+1])

        # Update the bond 
        f = tf.einsum('lmnik,tmnik->tl', bond, C)
        #f = tf.Print(f, [counter, f], message = 'counter, f')
        gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)
        label_bond = self.rate_of_change * gradient
        cutoff = 10**(-5)
        label_bond = tf.clip_by_value(label_bond, -(cutoff), cutoff)
        #label_bond = tf.Print(label_bond, [label_bond], message = "label_bond")
        random_part = tf.random_uniform(tf.shape(label_bond), minval=-(cutoff/10), maxval = cutoff/10)
        label_bond = tf.add(label_bond, random_part)
        updated_bond = tf.add(bond, label_bond)

        # Decompose the bond 
        aj, aj1 = self._bond_decomposition(updated_bond, self.bond_dim)


        # Transpose the values and add to the new variables 
        updated_nodes = updated_nodes.write(counter, aj)
        contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
        C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
        C1s = C1s.write(counter, C1)
        updated_counter = counter+1 

        return [updated_counter, C1s, C2s, updated_nodes, nodes, aj1]


    def _one_sweep(self, C1, C2, C2s):
        C1s = tf.TensorArray(tf.float32, size=self.MPS.input_size-2, dynamic_size = True, infer_shape=False, clear_after_read = False)
        n1 = self.MPS.nodes.read(1)
        n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])


        new_nodes = self._make_new_nodes(self.MPS.nodes)
        wrapped = [0, C1, C1s, C2s, new_nodes, self.MPS.nodes, n1]
        cond = lambda counter, b, c, d, e, f, g: tf.less(counter, self.MPS.input_size - 3)

        _, C1, C1s, C2s, self.updated_nodes, _, n1 = tf.while_loop(cond=cond, body=self._update, loop_vars=wrapped,
                                                                    parallel_iterations = 1,
                                                                    shape_invariants=[tf.TensorShape([]),
                                                                                      tf.TensorShape([None, None]),  
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape([None, None, None, None])])
        n1 = tf.transpose(n1, perm=[0, 1, 3, 2])
        self.updated_nodes = self.updated_nodes.write(1, n1)
        return (C1s, C2, C1)

    def _make_new_nodes(self, nodes):
        size = nodes.size()
        new_nodes = tf.TensorArray(tf.float32, size=size, dynamic_size=True, infer_shape=False, clear_after_read=False)
        new_nodes = new_nodes.write(0, nodes.read(size-1))
        new_nodes = new_nodes.write(size-1, nodes.read(0))
        return new_nodes

    def _duplicate_nodes(self, nodes, from_index, to_index):
        size = nodes.size()
        new_nodes = tf.TensorArray(tf.float32, size=size, dynamic_size=True, infer_shape=False, clear_after_read=False)
        cond = lambda index, a, b: tf.less(index, to_index)
        from_index = tf.cond(tf.equal(from_index, 0), lambda: 1, lambda: from_index)
        to_index = tf.cond(tf.greater(to_index, size - 1), lambda: size - 1, lambda: to_index)
        _, _, new_nodes  = tf.while_loop(cond = cond, body = self._transfer_to_array, loop_vars = [from_index, nodes, new_nodes])
        new_nodes = new_nodes.write(0, nodes.read(0))
        new_nodes = new_nodes.write(size-1, nodes.read(size-1))
        return new_nodes

    def _transfer_to_array(self, index, old_nodes, new_nodes):
        old_node = old_nodes.read(index)
        new_nodes = new_nodes.write(index, old_node)
        index += 1
        return (index, old_nodes, new_nodes)

    def _find_C2(self, counter, prev_C2, C2s):
        
        loc2 = self.MPS.input_size - 2 - counter
        node2 = self.MPS.nodes.read(loc2)
        node2.set_shape([self.MPS.d_feature, None, None])
        contracted_node2 = tf.einsum('mij,tm->tij', node2, self._feature[loc2]) # CHECK einsum
        updated_counter = counter + 1
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(self.MPS.input_size-5-counter, new_C2)
        return [updated_counter, new_C2, C2s]
        
        
    def _update(self, counter, C1, C1s, C2s, updated_nodes, nodes, previous_node):

        # Read in the notes 
        n1 = previous_node
        n2 = nodes.read(counter+2)
        n2.set_shape([self.MPS.d_feature, None, None])

        # Calculate the bond 
        bond = tf.einsum('lmij,njk->lmnik', n1, n2)

        # Calculate the C matrix 
        C2 = C2s.read(counter)
        C2.set_shape([None,None])
        C1s = C1s.write(counter, C1)
        C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, self._feature[counter], self._feature[counter+1])

        # Update the bond 
        f = tf.einsum('lmnik,tmnik->tl', bond, C)
        #f = tf.Print(f, [counter, f], message = 'counter, f')
        gradient = tf.einsum('tl,tmnik->lmnik', self._label-f, C)
        label_bond = self.rate_of_change * gradient
        cutoff = 10**(-5)
        label_bond = tf.clip_by_value(label_bond, -(cutoff), cutoff)
        label_bond = tf.Print(label_bond, [label_bond], message = "label_bond")
        random_part = tf.random_uniform(tf.shape(label_bond), minval=-(cutoff/10), maxval = cutoff/10)
        label_bond = tf.add(label_bond, random_part)
        updated_bond = tf.add(bond, label_bond)

        # Decompose the bond 
        aj, aj1 = self._bond_decomposition(updated_bond, self.bond_dim)


        # Transpose the values and add to the new variables 
        aj = tf.transpose(aj, perm=[0, 2, 1])
        updated_nodes = updated_nodes.write(self.MPS.input_size-2-counter, aj)
        contracted_aj = tf.einsum('mij,tm->tij', aj, self._feature[counter])
        C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
        updated_counter = counter+1 

        return [updated_counter, C1, C1s, C2s, updated_nodes, nodes, aj1]

    def _bond_decomposition(self, bond, m):
        """
        Decomposes bond, so that the next step can be done.
        :param bond:
        :param m:
        :return:
        """
        with tf.name_scope("bond_decomposition"):
            bond_reshaped = tf.transpose(bond, perm=[3, 1, 2, 0, 4])
            dims = tf.shape(bond_reshaped)
            l_dim = dims[0] * dims[1]
            r_dim = dims[2] * dims[3] * dims[4]
            bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
            s, u, v = tf.svd(bond_flattened)
            
            filtered_s = s
            s_size = tf.size(filtered_s)
            s_im = tf.reshape(tf.diag(filtered_s), [s_size, s_size, 1])
            v_im = tf.reshape(v, [s_size, r_dim, 1])
            u_im = tf.reshape(u, [l_dim, s_size, 1])
            s_im_cropped = tf.image.resize_image_with_crop_or_pad(s_im, m, m)
            v_im_cropped = tf.image.resize_image_with_crop_or_pad(v_im, m, r_dim)
            u_im_cropped = tf.image.resize_image_with_crop_or_pad(u_im, l_dim, m)
            s_mat = tf.reshape(s_im_cropped, [m, m])
            v_mat = tf.reshape(v_im_cropped, [m, r_dim])
            a_prime_j_mixed = tf.reshape(u_im_cropped, [dims[0], dims[1], m])
            sv = tf.matmul(s_mat, v_mat)
            a_prime_j1_mixed = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
            a_prime_j = tf.transpose(a_prime_j_mixed, perm=[1, 0, 2])
            a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[2, 1, 0, 3])

        return (a_prime_j, a_prime_j1)

if __name__ == '__main__':
    # Model parameters
    input_size = 196
    d_feature = 2
    d_matrix = 5
    d_output = 10
    rate_of_change = 10
    batch_size = 1000
    bond_dim = 5

    data_source = preprocessing.MNISTData()

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size)
    feature, labels = data_source.next_training_data_batch(batch_size)
    #network.test(feature, labels)
    optimizer = MPSOptimizer(network, bond_dim, None, rate_of_change = None)
    optimizer.train(data_source, batch_size)




    