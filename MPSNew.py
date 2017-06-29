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
        self.nodes = []

        # Initialise the nodes, input and output
        self._setup_nodes()

    def _setup_nodes(self):

        self.nodes = []
        # First node
        self.nodes.append(self._make_random_normal([self.d_feature, self.d_matrix]))
        # The Second node with output leg attached
        self.nodes.append(self._make_random_normal([self.d_output, self.d_feature, self.d_matrix, self.d_matrix]))
        # The rest of the matrix nodes
        for i in range(self.input_size - 2):
            self.nodes.append(self._make_random_normal([self.d_feature, self.d_matrix, self.d_matrix]))
        # Last node
        self.nodes.append(self._make_random_normal([self.d_feature, self.d_matrix]))

    def _make_random_normal(self, shape, mean=0, stddev=1):
        return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))

    def predict(self, input):
        """

        :param input:
        :return:
        """

def _node_at(index, nodes):
    #for node in nodes:
    #    print(node.shape)
    null_result = tf.constant(0.0)
    input_size = len(nodes)
    end_nodes, second_node, middle_nodes = _split(nodes)
    index = tf.cond(index < 0, lambda: tf.add(input_size, index), lambda: index)
    result = tf.cond(tf.equal(index,input_size - 1), lambda: end_nodes[index], lambda: null_result)
    result = tf.cond(tf.equal(index,1), lambda: second_node, lambda: result)
    result =  tf.cond(tf.equal(index,0), lambda: end_nodes[0], lambda: result)
    result = tf.cond(tf.equal(result, null_result), lambda: middle_nodes[index - 2], lambda: result)
    result = tf.cond(tf.equal(result, result), lambda: result, lambda: result)
    return result

def _split(nodes):
    end_nodes = []
    middle_nodes =[]
    second_node = []
    length = len(nodes)
    for index, element in enumerate(nodes):
        if index == 0 or index >= length -1:
            end_nodes.append(element)
        elif index == 1:
            second_node = element
        else:
            middle_nodes.append(element)
    end_nodes = tf.stack(end_nodes)
    middle_nodes = tf.stack(middle_nodes)
    return (end_nodes, second_node, middle_nodes)

class MPSOptimizer(object):
    def __init__(self, MPSNetwork, m, loss_func, rate_of_change = None):
        self.MPS = MPSNetwork
        if rate_of_change is None:
            self.rate_of_change = 0.1
        else:
            self.rate_of_change = rate_of_change
        self.m = m
        self.loss_func = loss_func
        self._phi = tf.placeholder(tf.float32, shape=[input_size, None, self.MPS.d_feature])
        self._delta = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])
        self._setup_optimization()
        self._setup_training_graph()


    def train(self, phi, delta):
        self._phi = phi
        self._delta = delta
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            end_results = sess.run(self.result,self.middle_nodes, {self.phi: phi, self.delta: delta})
            writer = tf.summary.FileWriter("output", sess.graph)
            writer.close()
        #self.MPS.nodes = end_results[-1]

    def _setup_optimization(self):
        phi = self._phi
        nodes = self.MPS.nodes
        self.C_1 = tf.einsum('ni,tn->ti', nodes[0], phi[0])
        C_2_1 = tf.einsum('mi,tm->ti', nodes[-1], phi[-1])
        C_2 = tf.stack([C_2_1, C_2_1])
        cond = lambda counter,b,c,d,e: tf.less(counter, self.MPS.input_size - 3)
        shape = []
        for _ in nodes:
            shape.append(tf.TensorShape(None))
        _, _, _, _, self.C_2= tf.while_loop(cond = cond, body = _find_C_2, loop_vars = [0.0, nodes, phi, C_2_1, C_2],
                                          shape_invariants = [tf.TensorShape([]), shape,
                                                              tf.TensorShape([self.MPS.input_size, None, self.MPS.d_feature]),
                                                              tf.TensorShape([None, None]), tf.TensorShape(None)])

    def _setup_training_graph(self):
       counter = 0
       phi = self._phi
       delta = self._delta
       updated_nodes = tf.TensorArray(tf.float32, size = 0, dynamic_size = True, infer_shape = False)
       wrapped = [counter, self.MPS.nodes, self._phi,
                  self._delta, self.C_1, self.C_2,
                  self.rate_of_change, updated_nodes, self.MPS.nodes[1]]
       cond = lambda counter, b, c, d, e, f, g, h, i: tf.less(counter, self.MPS.input_size - 2)
       shape = []
       for _ in self.MPS.nodes:
           shape.append(tf.TensorShape(None))
       self.result = tf.while_loop(cond= cond, body = _update, loop_vars=wrapped,
                                   shape_invariants = [tf.TensorShape([]), shape, tf.TensorShape([self.MPS.input_size, None, self.MPS.d_feature]),
                                                       self._delta.shape, tf.TensorShape([None,None]), self.C_2.shape,
                                                       tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape([None, None, None, None])])
       self.middle_nodes = self.result[-2].stack()

def _update(counter, nodes, phi, delta, C_1, C_2, rate, updated_nodes, previous_node):
    n1 = previous_node
    n2 = _node_at(counter+2, nodes)
    n1.set_shape([None, None, None, None])
    n2.set_shape([None, None, None])
    bond = tf.einsum('abcd,ecg->abedg',n1, n2)
    RHS = C_2[counter]
    RHS.set_shape([None,None])
    C = tf.einsum('ti,tk,tm,tn->tmnik', C_1, RHS, phi[counter], phi[counter+1])
    f = tf.einsum('lmnik,tmnik->tl', bond, C)
    # TODO: Change to delta  - f squared?
    gradient = tf.einsum('tl,tmnik->lmnik', delta - f, C)
    delta_bond = tf.scalar_mul(rate, gradient)
    updated_bond = tf.add(bond, delta_bond)
    aj, aj1 = _bond_decomposition(updated_bond, m)
    updated_nodes.write(-1, n1)
    updated_counter = tf.add(counter, 1)
    P = tf.einsum('flr,bf->lrb', aj, phi[counter])
    C_1 = tf.einsum('lrb,bl->br', P, C_1)
    wrapped = [updated_counter, nodes, phi, delta, C_1, C_2, rate, updated_nodes, aj1]
    return wrapped


def _find_C_2(counter, nodes, phi, prev_C2, results):
    loc2= tf.cast(-2 - counter, tf.int32)
    node2 = _node_at(loc2, nodes)
    node2.set_shape([phi[loc2].shape[1], None, None])
    processed_node2 = tf.einsum('abc,da->bcd', node2, phi[loc2])
    updated_counter = counter + 1
    C_2 = tf.einsum('acb,cb->ab',
                         processed_node2, prev_C2)
    results = tf.concat([results,C_2], 0)
    wrapped = [updated_counter, nodes, phi, C_2, results]
    return wrapped

def _bond_decomposition(bond, m):
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
    s_size = tf.Variable(tf.size(filtered_s))
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
    input_size = 10
    d_feature = 2
    d_matrix = 5
    d_output = 3
    rate_of_change = 0.2
    batch_size = 2
    m = 5

    # Make up input and output
    phi = np.random.normal(size=(input_size, batch_size, d_feature))
    delta = [[1, 0, 0], [0, 1, 0]]

    # Initialise the model
    network = MPS(d_matrix, d_feature, d_output, input_size)
    optimizer = MPSOptimizer(network, m, None, rate_of_change=rate_of_change)
    optimizer.train(phi,delta)
