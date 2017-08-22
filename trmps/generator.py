import tensorflow as tf
import numpy as np
from distribution import Quadratic
from mps import MPS
import pickle 
import utils
from matplotlib import pyplot as plt

class MPSGenerator(object):

    def __init__(self, MPSNetwork):
        self.MPS = MPSNetwork 
        assert self.MPS._special_node_loc == 0
        assert self.MPS.d_feature == 2

    def generate(self, n_samples, digit):
        samples_ta = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=True, clear_after_read=False)
        self.digit = digit
        self.n_samples = n_samples

        L = tf.transpose(self.MPS.start_node) * self.MPS.start_node
        L = tf.expand_dims(L, 0)
        L = tf.tile(L, [n_samples, 1, 1])
        cond = lambda c, l, t: tf.less(c, self.MPS.input_size)
        _, L, samples_ta = tf.while_loop(cond=cond, 
                             body=self._generate_from_one_node, 
                             loop_vars=[0, L, samples_ta],
                             shape_invariants=[tf.TensorShape([]), 
                                               tf.TensorShape([n_samples, None, None]), 
                                               tf.TensorShape(None)], 
                             parallel_iterations=5)
        R = tf.transpose(self.MPS.end_node) * self.MPS.end_node
        probs = tf.einsum('tij,ij->t', L, R)
        return samples_ta.stack(), probs
    
    def _sample_from_vector(self, vector):
        with tf.name_scope("sample_from_vector"):
            # sqrt the vector 
            vector_sign = tf.sign(vector) 
            vector = vector_sign * tf.sqrt(vector_sign * vector)

            vector = tf.Print(vector, [vector])
            vector = vector/tf.norm(vector) 

            dist = Quadratic(vector[0], vector[1])
            samples = dist.sample(self.n_samples)
            del dist 

        return samples


    def _generate_from_one_node(self, counter, L, samples_ta):

        with tf.name_scope("read_node"):
            node = self.MPS.nodes.read(counter)
            node = tf.cond(tf.equal(counter, self.MPS._special_node_loc), 
                           true_fn=lambda: self._normalise_special_node(node)[self.digit],
                           false_fn=lambda: node)


        with tf.name_scope("sample_from_node"):
            node.set_shape([self.MPS.d_feature, None, None])
            L_dot_node = tf.einsum('tij,mik,njk->mn', L, node, node)
            vector = tf.diag_part(L_dot_node)
            samples = self._sample_from_vector(vector)
            samples_ta = samples_ta.write(counter, samples) 

        with tf.name_scope("updated_L"):
            ones = tf.ones_like(samples) 
            feature = tf.stack([ones, np.sqrt(3) * (2 * samples - 1)], axis=1)

            contracted_node = tf.einsum('mij,tm->tij', node, feature)
            L = tf.einsum('tij,tik,tjl->tkl', L, contracted_node, contracted_node)

        return counter+1, L, samples_ta

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

    digit = 8
    n_samples = 5
    samples, probs = generator.generate(n_samples, digit)


    feed_dict = network.create_feed_dict(weights)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        samples, probs, norm = sess.run([samples, probs, norm], feed_dict=feed_dict)
        print(norm)
        print(probs)
        one_sample = samples[:, 0]
        utils.show(one_sample)
        plt.show()















