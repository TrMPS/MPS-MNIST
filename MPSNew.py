import tensorflow as tf
import numpy as np
from collections import namedtuple

#TODO: REALLY needs some documentation

class MPS(object):
    def __init__(self, d_matrix, d_feature, d_output, input_size):
        # structure parameters
        self.input_size = input_size
        self.d_matrix = d_matrix
        self.d_feature = d_feature
        self.d_output = d_output

        # Initialise the nodes, input and output
        self._setup_nodes()

    def _setup_nodes(self):

        self.nodes = tf.TensorArray(tf.float32, size = 0, dynamic_size= True,
                                    clear_after_read= False, infer_shape= False)
        # First node
        self.nodes = self.nodes.write(0, self._make_random_normal([self.d_feature, self.d_matrix]))
        # The Second node with output leg attached
        self.nodes = self.nodes.write(1, self._make_random_normal([self.d_output, self.d_feature, self.d_matrix, self.d_matrix]))
        # The rest of the matrix nodes

        for i in range(self.input_size - 3):
            self.nodes = self.nodes.write(i+2, self._make_random_normal([self.d_feature, self.d_matrix, self.d_matrix]))
        # Last node
        self.nodes = self.nodes.write(self.input_size - 1, self._make_random_normal([self.d_feature, self.d_matrix]))

    def _make_random_normal(self, shape, mean=0, stddev=1):
        return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))

    def predict(self, feature):
        phi = tf.placeholder(tf.float32, shape=[self.input_size, None, self.d_output])
        f = self._predict(phi)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(f, {phi: feature}))
            
    def _predict(self, phi):
        """
        :param phi:
        :return:
        """
        # Read in the nodes 
        node1 = self.nodes.read(0)
        node1.set_shape([self.d_feature, None])
        node2 = self.nodes.read(1)
        node2.set_shape([self.d_output, self.d_feature, None, None])
        nodelast = self.nodes.read(self.input_size-1)
        nodelast.set_shape([self.d_feature, None])

        # Calculate C1 
        C1 = tf.einsum('ni,tn->ti', node1, phi[0])
        contracted_node2 = tf.einsum('lnij,tn->tlij', node2, phi[1])
        C1 = tf.einsum('ti,tlij->tlj', C1, contracted_node2)

        # Calculate C2
        C2 = tf.einsum('mi,tm->ti', nodelast, phi[self.input_size-1])

        #counter = tf.Variable(2, dtype=tf.int32)
        cond = lambda counter, b: tf.less(counter, self.input_size-1)
        _, C1 = tf.while_loop(cond=cond, body=self._chain_multiply, loop_vars=[2, C1], 
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, self.d_output, None])])
        f = tf.einsum('tli,ti->tl', C1, C2)
        return f 

    def _chain_multiply(self, counter, C1):
        node = self.nodes.read(counter)
        node.set_shape([self.d_feature, None, None])
        print(counter.dtype)
        input_leg = phi[counter]
        contracted_node = tf.einsum('mij,tm->tij', node, input_leg)
        C1 = tf.einsum('tli,tij->tlj', C1, node)
        counter = counter + 1 
        return [counter, C1]






class MPSOptimizer(object):
    def __init__(self, MPSNetwork, m, grad_func, rate_of_change = None):
        self.MPS = MPSNetwork
        if rate_of_change is None:
            self.rate_of_change = 0.1
        else:
            self.rate_of_change = rate_of_change
        self.m = m
        self.grad_func = grad_func
        self._phi = tf.placeholder(tf.float32, shape=[input_size, None, self.MPS.d_feature])
        self._delta = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self._setup_optimization()
        self._setup_training_graph()


    def train(self, phi, delta):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            end_results = sess.run(self.C1, {self._phi: phi, self._delta: delta})
            print(end_results)
            writer = tf.summary.FileWriter("output", sess.graph)
            writer.close()
        #self.MPS.nodes = end_results[-1]

    def _setup_optimization(self):
        phi = self._phi
        nodes = self.MPS.nodes

        n1 = nodes.read(0)
        n1.set_shape([None,None])
        nlast = nodes.read(nodes.size() - 1)
        nlast.set_shape([None,None])

        C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size-3, infer_shape=False)
        self.C1 = tf.einsum('ni,tn->ti', n1, phi[0])
        C2 = tf.einsum('mi,tm->ti', nlast, phi[-1])
        C2s = C2s.write(self.MPS.input_size-4, C2)
        cond = lambda counter,b,c: tf.less(counter, self.MPS.input_size-4)
        # C2_finder = self._generate_C2_finder(nodes,phi)
        _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[0, C2, C2s], 
                                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]), 
                                        tf.TensorShape(None)])

    def _setup_training_graph(self):
       counter = 0
       updated_nodes = tf.TensorArray(tf.float32, size=self.MPS.input_size, dynamic_size=True, infer_shape=False)
       updated_nodes = updated_nodes.write(0, self.MPS.nodes.read(0))
       update_func = self._generate_update_func(self.MPS.nodes, self._phi, self._delta, self.rate_of_change)
       n1 = self.MPS.nodes.read(1)
       n1.set_shape([None, None, None, None])
       wrapped = [counter, self.C1, self.C2s,
                 updated_nodes, n1]
       cond = lambda counter, b, c, d, e: tf.less(counter, self.MPS.input_size - 2)
       _, self.C1, self.C2s, self.updated_nodes, _ = tf.while_loop(cond= cond, body = update_func, loop_vars=wrapped,
                                   shape_invariants = [tf.TensorShape([]),  tf.TensorShape([None,None]), tf.TensorShape(None),
                                                        tf.TensorShape(None), tf.TensorShape([None, None, None, None])])

    
    def _find_C2(self, counter, prev_C2, C2s):

        loc2 = self.MPS.input_size - 2 - counter
        node2 = self.MPS.nodes.read(loc2)
        node2.set_shape([self.MPS.d_feature, None, None])
        contracted_node2 = tf.einsum('mij,tm->tij', node2, self._phi[loc2]) # CHECK einsum
        updated_counter = counter + 1
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(self.MPS.input_size-5-counter, new_C2)
        return [updated_counter, new_C2, C2s]
        

    def _generate_update_func(self, nodes, phi, delta, rate):
        
        def _update(counter, C1, C2s, updated_nodes, previous_node):

            # Read in the notes 
            n1 = previous_node
            n2 = nodes.read(counter+2)
            n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the bond 
            bond = tf.einsum('lmij,njk->lmnik', n1, n2)

            # Calculate the C matrix 
            C2 = C2s.read(counter)
            C2.set_shape([None,None])
            C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, phi[counter], phi[counter+1])

            # Update the bond 
            f = tf.einsum('lmnik,tmnik->tl', bond, C)
            gradient = tf.einsum('tl,tmnik->lmnik', delta - f, C)
            delta_bond = rate * gradient
            updated_bond = tf.add(bond, delta_bond)

            # Decompose the bond 
            aj, aj1 = self._bond_decomposition(updated_bond, m)

            # Update and return the values
            updated_nodes = updated_nodes.write(counter, aj)
            contracted_aj = tf.einsum('mij,tm->tij', aj, phi[counter])
            C1 = tf.einsum('tij,ti->tj', contracted_aj, C1)
            updated_counter = counter + 1 
            return [updated_counter, C1, C2s, updated_nodes, aj1]
    
        return _update

    def _bond_decomposition(self, bond, m):
        """
        :param bond:
        :param m:
        :return:
        """
        bond_reshaped = tf.transpose(bond, perm=[3, 1, 2, 0, 4])
        bond_flattened = tf.reshape(bond_reshaped, [10, 30])
        s, u, v = tf.svd(bond_flattened)
        # filtered_s = s[:,-1]
        filtered_s = s
        s_size = tf.size(filtered_s)
        s_im = tf.reshape(tf.diag(filtered_s), [s_size, s_size, 1])
        v_im = tf.reshape(v, [s_size, 30, 1])
        u_im = tf.reshape(u, [10, s_size, 1])
        s_im_cropped = tf.image.resize_image_with_crop_or_pad(s_im, m, m)
        v_im_cropped = tf.image.resize_image_with_crop_or_pad(v_im, m, 30)
        u_im_cropped = tf.image.resize_image_with_crop_or_pad(u_im, 10, m)
        s_mat = tf.reshape(s_im_cropped, [m, m])
        v_mat = tf.reshape(v_im_cropped, [m, 30])
        a_prime_j_mixed = tf.reshape(u_im_cropped, [5, 2, m])
        sv = tf.matmul(s_mat, v_mat)
        a_prime_j1_mixed = tf.reshape(sv, [m, 2, 3, 5])
        a_prime_j = tf.transpose(a_prime_j_mixed, perm=[1, 0, 2])
        a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[3, 1, 0, 2])
        return (a_prime_j, a_prime_j1)

if __name__ == '__main__':
    # Model parameters
    input_size = 2
    d_feature = 5
    d_matrix = 5
    d_output = 3
    rate_of_change = 0.2
    batch_size = 10
    m = 5

    # Make up input and output
    phi = np.random.normal(size=(input_size, batch_size, d_feature))
    delta = [[1, 0, 0], [0, 1, 0]]

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size)
    network.predict(phi)
    