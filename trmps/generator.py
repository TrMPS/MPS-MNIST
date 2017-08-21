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

        L = tf.diag(tf.squeeze(self.MPS.start_node))
        L = tf.expand_dims(L, 0)
        L = tf.tile(L, [n_samples, 1, 1])
        cond = lambda c, l, t: tf.less(c, self.MPS.input_size)
        _, _, samples_ta = tf.while_loop(cond=cond, 
                             body=self._generate_from_one_node, 
                             loop_vars=[0, L, samples_ta],
                             shape_invariants=[tf.TensorShape([]), 
                                               tf.TensorShape([n_samples, None, None]), 
                                               tf.TensorShape(None)], 
                             parallel_iterations=5)
        return samples_ta.stack()
    
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
                           true_fn=lambda: node[self.digit],
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

    digit = 1
    n_samples = 10 
    samples = generator.generate(n_samples, digit)
    feed_dict = network.create_feed_dict(weights)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        samples = sess.run(samples, feed_dict=feed_dict)
        one_sample = samples[:, 0]
        utils.show(one_sample)
        plt.show()















