import tensorflow as tf
import numpy as np
from distribution import Quadratic
from mps import MPS
import pickle 
import utils
from matplotlib import pyplot as plt
import MNISTpreprocessing

class MPSGenerator(object):

    def __init__(self, MPSNetwork):
        self.MPS = MPSNetwork 
        assert self.MPS._special_node_loc == 0
        assert self.MPS.d_feature == 2

    def generate(self, n_samples, digit):
        samples_ta = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=True, clear_after_read=False)
        self.digit = digit
        self.n_samples = n_samples

        C1 = self.MPS.start_node
        C1 = tf.tile(C1, [n_samples, 1])
        cond = lambda c, l, t: tf.less(c, self.MPS.input_size)
        _, C1, samples_ta = tf.while_loop(cond=cond, 
                             body=self._generate_from_one_node, 
                             loop_vars=[0, C1, samples_ta],
                             shape_invariants=[tf.TensorShape([]), 
                                               tf.TensorShape([n_samples, None]), 
                                               tf.TensorShape(None)], 
                             parallel_iterations=5)
        C2 = tf.squeeze(self.MPS.end_node)
        probs = tf.einsum('ti,i->t', C1, C2)
        return samples_ta.stack(), probs
    
    def _sample_from_vector(self, vector):
        with tf.name_scope("sample_from_vector"):
            # sqrt the vector 
            vector = tf.Print(vector, [vector[0]])
            vector = vector/tf.norm(vector, axis=1, keep_dims=True) 

            dist = Quadratic(vector[:, 0], vector[:, 1])
            samples = dist.sample()
            del dist 

        return samples

    def _matrix_to_vector(self, matrices):
        '''
        Go from a matrix of vv^T to v
        '''
        vectors = tf.map_fn(lambda x:tf.diag_part(x), matrices)
        vectors = tf.sqrt(vectors)
        signs = tf.map_fn(lambda x:tf.sign(x[0]), matrices)
        return signs * vectors 

    def _generate_from_one_node(self, counter, C1, samples_ta):

        with tf.name_scope("read_node"):
            node = self.MPS.nodes.read(counter)
            node = tf.cond(tf.equal(counter, self.MPS._special_node_loc), 
                           true_fn=lambda: self._normalise_special_node(node)[self.digit],
                           false_fn=lambda: node)


        with tf.name_scope("sample_from_node"):
            node.set_shape([self.MPS.d_feature, None, None])
            C1_dot_node = tf.einsum('ti,mij->tmj', C1, node)
            contracted_C1_dot_node = tf.einsum('tmj,tnj->tmn', C1_dot_node, C1_dot_node)
            vector = self._matrix_to_vector(contracted_C1_dot_node)

            samples = self._sample_from_vector(vector)
            samples_ta = samples_ta.write(counter, samples) 

        with tf.name_scope("updated_L"):
            ones = tf.ones_like(samples) 
            feature = tf.stack([ones, np.sqrt(3) * (2 * samples - 1)], axis=1)

            C1 = tf.einsum('tmj,tm->tj', C1_dot_node, feature)

        return counter+1, C1, samples_ta

    def _normalise_special_node(self, special_node):
        special_node.set_shape([self.MPS.d_output, self.MPS.d_feature, None, None])
        matrix_dim_1 = tf.shape(special_node)[-2] 
        matrix_dim_2 = tf.shape(special_node)[-1]
        collective_dim = self.MPS.d_output * self.MPS.d_feature * matrix_dim_1
        reshaped_node = tf.reshape(special_node, [collective_dim, matrix_dim_2])
        s, u, v = tf.svd(reshaped_node)
        s = s/tf.norm(s) 
        u = u * tf.expand_dims(s, 0)
        special_node = tf.matmul(u, v, transpose_b=True)
        special_node = tf.reshape(special_node, 
                                  [self.MPS.d_output, self.MPS.d_feature, matrix_dim_1, matrix_dim_2])

        return special_node

    def _check_norm(self):

        def _multiply_right(counter, L):
            node = self.MPS.nodes.read(counter)
            node.set_shape([self.MPS.d_feature, None, None])
            T = tf.einsum('mij,mkl->ikjl', node, node)
            L = tf.einsum('ik,ikjl->jl', L, T)
            counter = counter + 1 

            return counter, L 

        special_node = self._normalise_special_node(self.MPS.nodes.read(0))
        L = tf.einsum('lnij,lnik->jk', special_node, special_node)
        cond = lambda c, l: tf.less(c, self.MPS.input_size)
        _, L = tf.while_loop(cond=cond, 
                             body=_multiply_right, 
                             loop_vars=[1, L],
                             shape_invariants=[tf.TensorShape([]), 
                                               tf.TensorShape([None, None])],
                             parallel_iterations=5)
        return tf.trace(L)

if __name__ == '__main__':

    # Model parameters
    input_size = 784
    shrink = True
    shuffled = True
    permuted = False
    special_node_loc = 0

    if shrink:
        input_size = 196
    d_feature = 2
    d_output = 10

    # Initialise the model

    with open('weights', 'rb') as fp:
        weights = pickle.load(fp)
        if len(weights) != input_size:
            weights = None

    # Initialise the model
    network = MPS(d_feature, d_output, input_size, special_node_loc)
    network.prepare(data_source=None)

    generator = MPSGenerator(network)
    norm = generator._check_norm()

    digit = 7
    n_samples = 100
    samples, probs = generator.generate(n_samples, digit)

    feature = tf.stack([tf.ones_like(samples), np.sqrt(3) * (2 * samples - 1)], axis=-1)
    label_np = np.zeros([n_samples, 10])
    label_np[:, digit] = 1 
    label = tf.constant(label_np)

    f = network.predict(feature)
    cost = network.cost(f, label)
    accuracy = network.accuracy(f, label)
    confusion = network.confusion_matrix(f, label)
        

    feed_dict = network.create_feed_dict(weights)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        to_eval = [samples, cost, accuracy, confusion]
        samples, cost, accuracy, confusion = sess.run(to_eval, feed_dict=feed_dict)
        print(cost, accuracy)
        print(confusion)
        one_sample = samples[:, 1]
        utils.show(one_sample)
        plt.show()




















