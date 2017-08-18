import tensorflow as tf
import numpy as np
from distribution import Quadratic
from mps import MPS
import pickle 

class MPSGenerator(object):

    def __init__(self, MPSNetwork):
        self.MPS = MPSNetwork 
        assert self.MPS._special_node_loc == 0
        assert self.MPS.d_feature == 2

    def generate(self, n_samples, digit):
        self.samples = tf.zeros([n, self.MPS.input_size], dtype=tf.float32)
        self.digit = digit
        self.n_samples = n_samples

        L = tf.tile(tf.diag(self.MPS.start_node), [n, 1, 1])
        cond = lambda c, l: tf.less(c, self.MPS.input_size)
        _, _ = tf.while_loop(cond=cond, 
                             body=self._generate_from_one_node, 
                             loop_vars=[0, L],
                             shape_invariants=[tf.TensorShape([]), tf.TensorShape([n_samples, None, None])], 
                             parallel_iterations=5)
        return self.samples
    
    def _sample_from_vector(self, vector):
        # sqrt the vector 
        vector_sign = tf.sign(vector) 
        vector = vector_sign * tf.sqrt(vector_sign * vector)

        vector = tf.Print(vector, [vector])
        vector = vector/tf.norm(vector) 

        dist = Quadratic(vector[0], vector[1])
        samples = dist.sample(self.n_samples)
        del dist 

        return samples


    def _generate_from_one_node(self, counter, L):
        node = self.MPS.nodes.read(counter)
        node = tf.cond(tf.equal(counter, self.MPS._special_node_loc), 
                       true_fn=lambda: node[self.digit],
                       false_fn=lambda: node)

        node.set_shape([self.d_feature, None, None])
        L_dot_node = tf.einsum('tij,mik,njk->mn', L, node, node)
        vector = tf.diag_part(L_dot_node)
        samples = self._sample_from_vector(vector)
        self.samples[:, counter] = samples 

        ones = tf.ones_like(samples) 
        feature = tf.stack([ones, np.sqrt(3) * (2 * samples - 1)], axis=1)

        contracted_node = tf.einsum('mij,tm->tij', node, feature)
        L = tf.einsum('tij,tik,tjl->tkl', L, contracted_node, contract_node)

        return counter+1, L

if __name__ == '__main__':

    # Model parameters
    input_size = 784
    shrink = True
    shuffled = True
    permuted = False

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
    network = MPS(d_feature, d_output, input_size)
    network.prepare(data_source=None)
    feed_dict = network.create_feed_dict(weights)

    generator = MPSGenerator(network)

    digit = 5 
    n_samples = 10 
    samples = MPSGenerator.generate(n_samples, digit)

    with tf.Session() as sess: 
        samples = sess.run(samples, feed_dict=feed_dict)













