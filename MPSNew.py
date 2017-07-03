import tensorflow as tf
import numpy as np
from collections import namedtuple


# TODO: REALLY needs some documentation

class MPS(object):
    def __init__(self, d_matrix, d_feature, d_output, input_size):
        # structure parameters
        # input_size must be > 4 for this to work
        self.input_size = input_size
        self.d_matrix = d_matrix
        self.d_feature = d_feature
        self.d_output = d_output
        self.nodes = []

        # Initialise the nodes, input and output
        self._setup_nodes()

    def _setup_nodes(self):
        self.nodes = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                    clear_after_read=False, infer_shape=False)
        # First node
        self.nodes = self.nodes.write(0, self._make_random_normal([self.d_feature, self.d_matrix]))
        # The Second node with output leg attached
        self.nodes = self.nodes.write(1, self._make_random_normal(
            [self.d_output, self.d_feature, self.d_matrix, self.d_matrix]))
        # The rest of the matrix nodes

        for i in range(self.input_size - 3):
            self.nodes = self.nodes.write(i + 2,
                                          self._make_random_normal([self.d_feature, self.d_matrix, self.d_matrix]))
        # Last node
        self.nodes = self.nodes.write(self.input_size - 1, self._make_random_normal([self.d_feature, self.d_matrix]))

    def _make_random_normal(self, shape, mean=0, stddev=1):
        return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))

    def predict(self, input):
        """

        :param input:
        :return:
        """


class MPSOptimizer(object):
    """
    Class to optimize an MPS system. Create this object, and then use train(phi, delta),
    where phi represents the inputs, and delta the correct labels.
    """
    def __init__(self, MPSNetwork, m, grad_func, rate_of_change=None):
        """
        :param MPSNetwork: MPS, the MPS network to be optimised.
        :param m: integer, the maximum size of the A matrices comprising the MPS
        :param grad_func: Some function which determines the direction in which the graph should change.
        (is derived by differentiating the loss function)
        :param rate_of_change: The rate of change for the network.
        """
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
        # TODO: rewrite this so it adds nodes to graph, which are then run outside, instead of current form.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            end_results = sess.run(self.C1, {self._phi: phi, self._delta: delta})
            print(end_results)
            writer = tf.summary.FileWriter("output", sess.graph)
            writer.close()
            # self.MPS.nodes = end_results[-1]

    def _setup_optimization(self):
        """
        Creates the initial values for C1, C2 (the left hand and right hand collected nodes/inputs)
        as a warmup to the optimisation.
        :return: nothing
        """
        phi = self._phi
        nodes = self.MPS.nodes

        n1 = nodes.read(0)
        n1.set_shape([None, None])
        nlast = nodes.read(nodes.size() - 1)
        nlast.set_shape([None, None])

        C2s = tf.TensorArray(tf.float32, size=self.MPS.input_size - 3, infer_shape=False)
        self.C1 = tf.einsum('ni,tn->ti', n1, phi[0])
        C2 = tf.einsum('mi,tm->ti', nlast, phi[-1])
        C2s = C2s.write(self.MPS.input_size - 4, C2)
        cond = lambda counter, b, c: tf.less(counter, self.MPS.input_size - 4)
        # C2_finder = self._generate_C2_finder(nodes,phi)
        _, _, self.C2s = tf.while_loop(cond=cond, body=self._find_C2, loop_vars=[0, C2, C2s],
                                       parallel_iterations = 1,
                                       shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]),
                                                         tf.TensorShape(None)])

    def _setup_training_graph(self):
        """
        Sets up graph needed to train the MPS, only for going to the right once.
        :return: nothing
        """
        counter = 0
        updated_nodes = tf.TensorArray(tf.float32, size=self.MPS.input_size, dynamic_size=True, infer_shape=False)
        updated_nodes = updated_nodes.write(0, self.MPS.nodes.read(0))
        update_func = self._generate_update_func(self.MPS.nodes, self._phi, self._delta, self.rate_of_change)
        n1 = self.MPS.nodes.read(1)
        n1.set_shape([None, None, None, None])
        wrapped = [counter, self.C1, self.C2s,
                   updated_nodes, n1]
        cond = lambda counter, b, c, d, e: tf.less(counter, self.MPS.input_size - 3)
        _, self.C1, self.C2s, self.updated_nodes, _ = tf.while_loop(cond=cond, body=update_func, loop_vars=wrapped,
                                                                    parallel_iterations = 1,
                                                                    shape_invariants=[tf.TensorShape([]),
                                                                                      tf.TensorShape([None, None]),
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape(None),
                                                                                      tf.TensorShape(
                                                                                          [None, None, None, None])])

    def _find_C2(self, counter, prev_C2, C2s):
        """
        Finds C2 values, used internally by _setup_optimisation
        :param counter:
        :param prev_C2:
        :param C2s:
        :return:
        """

        loc2 = self.MPS.input_size - 2 - counter
        node2 = self.MPS.nodes.read(loc2)
        node2.set_shape([self.MPS.d_feature, None, None])
        contracted_node2 = tf.einsum('mij,tm->tij', node2, self._phi[loc2])  # CHECK einsum
        updated_counter = counter + 1
        new_C2 = tf.einsum('tij,tj->ti', contracted_node2, prev_C2)
        C2s = C2s.write(self.MPS.input_size - 5 - counter, new_C2)
        return [updated_counter, new_C2, C2s]

    def _generate_update_func(self, nodes, phi, delta, rate):
        """
        Generates function required to setup the training graph. Used internally by _setup_training_graph.
        Currently broken.
        :param nodes:
        :param phi:
        :param delta:
        :param rate:
        :return:
        """

        def _update(counter, C1, C2s, updated_nodes, previous_node):
            # Read in the notes
            n1 = previous_node
            n2 = nodes.read(counter + 2)
            n1.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
            n2.set_shape([self.MPS.d_feature, None, None])

            # Calculate the bond
            # Seems to think (on nth time this loop is accessed, n>1)
            # that n2 has dimensions
            # input_size x matrix_size x output_shape
            # where it should be
            # input_size x matrix_size x matrix_size?
            # Most probably not due to the bond decomposition as that only
            # affects n1, have no idea why it's happening?
            bond = tf.einsum('lmij,njk->lmnik', n1, n2)

            # Calculate the C matrix
            C2 = C2s.read(counter)
            C2.set_shape([None, None])
            C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, phi[counter], phi[counter + 1])

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
        Decomposes bond, so that the next step can be done.
        :param bond:
        :param m:
        :return:
        """
        bond_reshaped = tf.transpose(bond, perm=[3, 1, 2, 0, 4])
        dims = tf.shape(bond_reshaped)
        l_dim = dims[0] * dims[1]
        r_dim = dims[2] * dims[3] * dims[4]
        bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
        s, u, v = tf.svd(bond_flattened)
        # filtered_s = s[:,-1]
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
        a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[3, 1, 0, 2])
        return (a_prime_j, a_prime_j1)


if __name__ == '__main__':
    # Model parameters
    input_size = 10
    d_feature = 2
    d_matrix = 5
    d_output = 6
    rate_of_change = 0.2
    batch_size = 15
    m = 10

    # Make up input and output
    phi = np.random.normal(size=(input_size, batch_size, d_feature))
    delta = []
    for i in range(batch_size):
        delta.append([0,1,0,0,0,0])

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size)
    optimizer = MPSOptimizer(network, m, None, rate_of_change=rate_of_change)
    optimizer.train(phi, delta)
