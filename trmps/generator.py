import tensorflow as tf
import numpy as np
from distribution import Quadratic
from shortMPS import *
import pickle
import utils
from matplotlib import pyplot as plt
import MNISTpreprocessing

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
        with tf.name_scope("recover_vectors"):
            vectors = tf.map_fn(lambda x:tf.diag_part(x), matrices)
            vectors = tf.sqrt(vectors)
            signs = tf.map_fn(lambda x:tf.sign(x[0]), matrices)
            vectors = signs * vectors

        return self._sample_from_vectors(vectors)

    def _sample_from_vectors(self, vectors):
        with tf.name_scope("sample_from_vectors"):
            # vectors = tf.Print(vectors, [vectors[0]])
            dist = Quadratic(vectors[:, 0], vectors[:, 1], tol=self._tol)
            samples = dist.sample()
            del dist

        return samples

    def _sample_from_node(self, counter, middle, samples_ta, right_flag):

        with tf.name_scope("read_node"):
            node = self.MPS.nodes.read(counter)
            node.set_shape([self.MPS.d_feature, None, None])

        with tf.name_scope("sample_from_node"):
            middle_dot_node = tf.cond(right_flag,
                                      true_fn=lambda: tf.einsum('tij,mjk->tmik', middle, node),
                                      false_fn=lambda: tf.einsum('tjk,mij->tmik', middle, node))
            contracted_node = tf.einsum('tmik,tnik->tmn', middle_dot_node, middle_dot_node)
            samples = self._sample_from_matrices(contracted_node)
            samples_ta = samples_ta.write(counter, samples)

        with tf.name_scope("update_middle"):
            feature = self._preprocess(samples)
            middle = tf.einsum('tmik,tm->tik', middle_dot_node, feature)

        with tf.name_scope("update_counter"):
            counter = tf.cond(right_flag,
                              true_fn=lambda: counter+1,
                              false_fn=lambda: counter-1)

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

    # Initialise the model

    with open('weights', 'rb') as fp:
        weights = pickle.load(fp)
        if len(weights) != input_size:
            print("weights not of desired shape")
            weights = None

    # Initialise the model
    network = shortMPS(d_feature, d_output, input_size, special_node_loc)
    network.prepare(data_source=None)

    generator = MPSGenerator(network)

    digit = 5
    n_samples = 500
    samples, pdfs = generator.generate(n_samples, digit, tol)

    feature = tf.stack([tf.ones_like(samples), np.sqrt(3) * (2 * samples - 1)], axis=-1)
    label_np = np.zeros([n_samples, 11])
    label_np[:, digit] = 1
    label = tf.constant(label_np)

    f = network.predict(feature)
    cost = network.cost(f, label)
    accuracy = network.accuracy(f, label)
    confusion = network.confusion_matrix(f, label)

    feed_dict = network.create_feed_dict(weights)


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
    plt.figure()
    utils.show(samples[:, 0])
    plt.show()




















