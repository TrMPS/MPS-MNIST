import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
from generation.samplers import quad_sample, higher_order_sample
from mps.shortMPS import *
import utils
from matplotlib import pyplot as plt
import preprocessing.MNISTpreprocessing as MNISTpreprocessing

class MPSGenerator(object):
    """
    A generator object that takes an MPS network
    and generate samples from it by unzipping the chain.

    It can generate from both MPS and shortMPS.

    Example usage using the MNIST dataset:

    import tensorflow as tf
    import numpy as np
    from shortMPS import *
    from generator import *
    import pickle
    import utils
    from matplotlib import pyplot as plt
    import MNISTpreprocessing

    # Initialise a shortMPS

    input_size = 784
    shrink = True
    shuffled = True
    permuted = False
    special_node_loc = 91

    if shrink:
        input_size = 196
    d_feature = 2
    d_output = 10

    network = shortMPS(d_feature, d_output, input_size, special_node_loc)
    network.prepare(data_source=None)

    # Load the weights

    with open('weights', 'rb') as fp:
    weights = pickle.load(fp)
    if len(weights) != input_size:
        print("weights not of desired shape")
        weights = None

    feed_dict = network.create_feed_dict(weights)



    # Generate samples

    generator = MPSGenerator(network)

    tol = 1e-3
    digit = 1
    n_samples = 500
    samples, pdfs = generator.generate(n_samples, digit, tol)

    feature = tf.stack([tf.ones_like(samples), np.sqrt(3) * (2 * samples - 1)], axis=-1)
    label_np = np.zeros([n_samples, 10])
    label_np[:, digit] = 1
    label = tf.constant(label_np)

    # Test the generated samples

    f = network.predict(feature)
    cost = network.cost(f, label)
    accuracy = network.accuracy(f, label)
    confusion = network.confusion_matrix(f, label)

    # Run tensorflow sesion

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        to_eval = [samples, pdfs, cost, accuracy, confusion]
        samples, pdfs, cost, accuracy, confusion = sess.run(to_eval, feed_dict=feed_dict)
        print('Cost: ', cost, ', Accuracy: ', accuracy)
        print(confusion)
        print('Pixels with values larger than 1: ')
        print(samples[samples > 1])

    print(samples.shape)
    utils.show(samples[:, np.argmax(pdfs)])
    plt.figure()
    avg_samples = np.mean(samples, axis=1)
    utils.show(avg_samples)
    plt.show()
    """

    def __init__(self, MPSNetwork):
        """
        :param MPSNetwork: MPS
            The matrix product state network that will be optimised
        :return:
        """
        self.MPS = MPSNetwork
        assert self.MPS.d_feature == 2

    def generate(self, n_samples, digit, tol=1e-6):
        """
        :param n_samples: integer
            The number of samples required
        :param digit: integer
            The integer we want to sample
        :param tol: float
            The tolerance for the Quadratic distribution
        :return:
        """
        self.samples_ta = tf.TensorArray(tf.float32, size=self.MPS.input_size, infer_shape=True, clear_after_read=False)
        self.digit = digit
        self.n_samples = n_samples
        self._tol = tol


        loc = self.MPS._special_node_loc
        middle = self._sample_from_special_node()
        cond = lambda c, l, t, f: tf.less(c, self.MPS.input_size)
        _, middle, self.samples_ta, _ = tf.while_loop(cond=cond,
                             body=self._sample_from_node,
                             loop_vars=[loc+1, middle, self.samples_ta, True],
                             shape_invariants=[tf.TensorShape([]),
                                               tf.TensorShape([None, None, None]),
                                               tf.TensorShape(None),
                                               tf.TensorShape([])],
                             parallel_iterations=5)


        cond = lambda c, l, t, f: tf.greater_equal(c, 0)
        _, middle, self.samples_ta, _ = tf.while_loop(cond=cond,
                                    body=self._sample_from_node,
                                    loop_vars=[loc-1, middle, self.samples_ta, False],
                                    shape_invariants=[tf.TensorShape([]),
                                                       tf.TensorShape([None, None, None]),
                                                       tf.TensorShape(None),
                                                       tf.TensorShape([])],
                                    parallel_iterations=5)
        pdfs = tf.squeeze(middle)

        return self.samples_ta.stack(), pdfs

    def _sample_from_matrices(self, matrices):
        # with tf.name_scope("recover_vectors"):
        #     vectors = tf.svd(matrices, compute_uv=False)
        #     # vectors = tf.map_fn(lambda x: tf.diag_part(x), matrices)
        #     # vectors = tf.sqrt(vectors)
        #     # signs = tf.map_fn(lambda x: tf.sign(x[0]), matrices)
        #     # vectors = signs * vectors

        # return self._sample_from_vectors(vectors)
        return self._new_sample_from_matrices(matrices)

    def _new_sample_from_matrices(self, matrices):
        samples = quad_sample(matrices, n=10)
        # samples = higher_order_sample(matrices, n=10)
        return samples

    def _sample_from_vectors(self, vectors):
        with tf.name_scope("sample_from_vectors"):
            # vectors = tf.Print(vectors, [vectors[0]])
            # dist = Quadratic(vectors[:, 0], vectors[:, 1], tol=self._tol)
            # samples = dist.sample()
            # del dist
            samples = higher_order_sample(vectors[:, 0], vectors[:, 1], n=10)

        return samples

    def _sample_from_node(self, counter, middle, samples_ta, right_flag):
        # middle = tf.Print(middle, [tf.shape(middle), right_flag, middle[0]], message='middle', summarize=1000)

        with tf.name_scope("read_node"):
            node = self.MPS.nodes.read(counter)
            node.set_shape([self.MPS.d_feature, None, None])

        with tf.name_scope("sample_from_node"):
            middle_dot_node = tf.cond(right_flag,
                                      true_fn=lambda: tf.einsum('tij,mjk->tmik', middle, node),
                                      false_fn=lambda: tf.einsum('tjk,mij->tmik', middle, node))
            contracted_node = tf.einsum('tmik,tnik->tmn', middle_dot_node, middle_dot_node)
            contracted_node = tf.Print(contracted_node, [right_flag, contracted_node[0]], message='contracted_node', summarize=100)
            samples = self._sample_from_matrices(contracted_node)
            samples_ta = samples_ta.write(counter, samples)

        with tf.name_scope("update_middle"):
            feature = self._preprocess(samples)
            middle = tf.einsum('tmik,tm->tik', middle_dot_node, feature)

        with tf.name_scope("update_counter"):
            counter = tf.cond(right_flag,
                              true_fn=lambda: counter + 1,
                              false_fn=lambda: counter - 1)

        return counter, middle, samples_ta, right_flag

    def _sample_from_special_node(self):
        loc = self.MPS._special_node_loc
        with tf.name_scope("normalise_special_node"):
            special_node = self.MPS.nodes.read(loc)
            special_node = self._normalise_special_node(special_node)

        with tf.name_scope("sample_from_special_node"):
            matrix = tf.einsum('mij,nij->mn', special_node[self.digit], special_node[self.digit])
            matrix = tf.expand_dims(matrix, 0)
            matrices = tf.tile(matrix, [self.n_samples, 1, 1])

            samples = self._sample_from_matrices(matrices)
            self.samples_ta = self.samples_ta.write(loc, samples)

        with tf.name_scope("attach_samples"):
            features = self._preprocess(samples)
            contracted_node = tf.einsum('mij,tm->tij', special_node[self.digit], features)

        return contracted_node


    def _preprocess(self, samples):
        ones = tf.ones_like(samples)
        feature = tf.stack([ones, np.sqrt(3) * (2 * samples - 1)], axis=-1)
        return feature


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

if __name__ == '__main__':

    # Model parameters
    input_size = 784
    shrink = True
    shuffled = True
    permuted = False
    special_node_loc = 91

    if shrink:
        input_size = 196
    d_feature = 2
    d_output = 11

    # Tolerance for generation
    tol = 1e-3

    # weights = None
    # data_source = MNISTpreprocessing.MNISTDatasource(shrink, permuted=permuted)

    # Initialise the model
    network = shortMPS.from_file()

    generator = MPSGenerator(network)

    digit = 3
    n_samples = 500
    samples, pdfs = generator.generate(n_samples, digit, tol)

    feature = tf.stack([tf.ones_like(samples), np.sqrt(3) * (2 * samples - 1)], axis=-1)
    label_np = np.zeros([n_samples, d_output])
    label_np[:, digit] = 1
    label = tf.constant(label_np)

    f = network.predict(feature)
    cost = network.cost(f, label)
    accuracy = network.accuracy(f, label)
    confusion = network.confusion_matrix(f, label)

    feed_dict = network.create_feed_dict(None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        to_eval = [samples, pdfs, cost, accuracy, confusion, f]
        samples, pdfs, cost, accuracy, confusion, prediction = sess.run(to_eval, feed_dict=feed_dict)
        print('sample prediction: ', prediction[0])
        print('average prediction: ', np.mean(prediction, axis=0))
        print('Cost: ', cost, ', Accuracy: ', accuracy)
        print(confusion)
        print('Pixels with values larger than 1: ')
        print(samples[samples > 1])

    print(samples.shape)
    utils.show(samples[:, np.argmax(pdfs)])
    plt.figure()
    avg_samples = np.mean(samples, axis=1)
    utils.show(avg_samples, normalise=True)
    plt.figure()
    utils.show(samples[:, 0])
    plt.figure()
    plt.hist(avg_samples)
    plt.xlim(0, 1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Occurences')
    plt.figure()
    plt.hist(samples[:, np.argmax(pdfs)])
    plt.xlim(0, 1)

    plt.show()




















