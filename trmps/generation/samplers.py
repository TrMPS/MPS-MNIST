import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generation.solvers import _bisection_solver


def quad_sample(matrices, n, name='quadratic', seed=None):
    a = matrices[:, 0, 0]
    b_c = matrices[:, 0, 1] + matrices[:, 1, 0]
    d = matrices[:, 1, 1]
    norm = (a + np.sqrt(3) * (b_c)
            - np.sqrt(3) * b_c
            + d * 4
            - d * 6
            + 3 * d)
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, _ = _generate_dist_funcs(a, b_c, d, ys * norm)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    return _bisection_solver(f=cdfs, x_dim=x_dim, n=n)


def higher_order_sample(matrices, n, name='quadratic', seed=None):
    a = matrices[:, 0, 0]
    b_c = matrices[:, 0, 1] + matrices[:, 1, 0]
    d = matrices[:, 1, 1]
    # norm = ((a + np.sqrt(3) * b_c + 3 * d)**3
    #         + 3 * (a + np.sqrt(3) * b_c + 3 * d)**2 * (np.sqrt(3) * b_c + 6 * d)
    #         + 12 * (np.sqrt(3) * b_c**3 + a**2 * d + 18 * b_c**2 * d + 30 * np.sqrt(3) * b_c * d**2 + 45 * d**3 + a * (b_c**2 + 6 * np.sqrt(3) * b_c * d + 18 * d**2))
    #         + 6 * (np.sqrt(3) * b_c + 6 * d) * (b_c**2 + 10 * np.sqrt(3) * b_c * d + 6 * d * (a + 5 * d))
    #         + (432 * d * (b_c**2 + 5 * np.sqrt(3) * b_c * d + d * (a + 15 * d))) / 5 + 144 * d**2 * (np.sqrt(3) * b_c + 6 * d)
    #         + (1728 * d**3) / 7)
    norm = (a ** 6 - 6 * a ** 5 * (np.sqrt(3) * b_c - 3 * d) + 45 * a ** 4 * (b_c ** 2 - 2 * np.sqrt(3) * b_c * d + 3 * d ** 2) - 60 * a ** 3 * (np.sqrt(3) * b_c ** 3 - 9 * b_c ** 2 * d + 9 * np.sqrt(3) * b_c * d ** 2 - 9 * d ** 3) + 135 * a ** 2 * (b_c ** 4 - 4 * np.sqrt(3) * b_c ** 3 * d + 18 * b_c ** 2 * d ** 2 - 12 * np.sqrt(3) * b_c * d ** 3 + 9 * d ** 4) - 54 * a * (np.sqrt(3) * b_c ** 5 - 15 * b_c ** 4 * d + 30 * np.sqrt(3) * b_c ** 3 * d ** 2 - 90 * b_c ** 2 * d ** 3 + 45 * np.sqrt(3) * b_c * d ** 4 - 27 * d ** 5) + 27 * (b_c ** 6 - 6 * np.sqrt(3) * b_c ** 5 * d + 45 * b_c ** 4 * d ** 2 - 60 * np.sqrt(3) * b_c ** 3 * d ** 3 + 135 * b_c ** 2 * d ** 4 - 54 * np.sqrt(3) * b_c * d ** 5 + 27 * d ** 6)) * 1 - 6 * (np.sqrt(3) * b_c - 6 * d) * (-a + np.sqrt(3) * b_c - 3 * d) ** 5 * 1 ** 2 + 12 * (a - np.sqrt(3) * b_c + 3 * d) ** 4 * (5 * b_c ** 2 - 22 * np.sqrt(3) * b_c * d + 2 * d * (a + 33 * d)) * 1 ** 3 + 60 * (np.sqrt(3) * b_c - 6 * d) * (-a + np.sqrt(3) * b_c - 3 * d) ** 3 * (-2 * b_c ** 2 + 11 * np.sqrt(3) * b_c * d - 3 * d * (a + 11 * d)) * 1 ** 4 + 432 * (a - np.sqrt(3) * b_c + 3 * d) ** 2 * (b_c ** 4 - 12 * np.sqrt(3) * b_c ** 3 * d - 18 * np.sqrt(3) * b_c * d ** 2 * (a + 11 * d) + b_c ** 2 * d * (4 * a + 135 * d) + d ** 2 * (a ** 2 + 54 * a * d + 297 * d ** 2)) * 1 ** 5 - 288 * (np.sqrt(3) * b_c - 6 * d) * (np.sqrt(3) * b_c ** 5 - 57 * b_c ** 4 * d - 10 * a ** 3 * d ** 2 + 306 * np.sqrt(3) * b_c ** 3 * d ** 2 - 1944 * b_c ** 2 * d ** 3 + 1782 * np.sqrt(3) * b_c * d ** 4 - 1782 * d ** 5 - 10 * a ** 2 * d * (b_c ** 2 - 7 * np.sqrt(3) * b_c * d + 21 * d ** 2) - a * (b_c ** 4 - 28 * np.sqrt(3) * b_c ** 3 * d + 462 * b_c ** 2 * d ** 2 - 756 * np.sqrt(3) * b_c * d ** 3 + 1134 * d ** 4)) * 1 ** 6 + (1728 * (b_c ** 6 - 42 * np.sqrt(3) * b_c ** 5 * d - 420 * np.sqrt(3) * b_c ** 3 * d ** 2 * (a + 12 * d) + 30 * b_c ** 4 * d * (a + 42 * d) - 84 * np.sqrt(3) * b_c * d ** 3 * (5 * a ** 2 + 90 * a * d + 297 * d ** 2) + 90 * b_c ** 2 * d ** 2 * (a ** 2 + 56 * a * d + 315 * d ** 2) + 4 * d ** 3 * (5 * a ** 3 + 315 * a ** 2 * d + 2835 * a * d ** 2 + 6237 * d ** 3)) * 1 ** 7)/7 - 2592 * (np.sqrt(3) * b_c - 6 * d) * d * (-b_c ** 4 + 18 * np.sqrt(3) * b_c ** 3 * d + 12 * np.sqrt(3) * b_c * d ** 2 * (5 * a + 33 * d) - 2 * b_c ** 2 * d * (5 * a + 126 * d) - 2 * d ** 2 * (5 * a ** 2 + 90 * a * d + 297 * d ** 2)) * 1 ** 8 + 34560 * d ** 2 * (b_c ** 4 - 12 * np.sqrt(3) * b_c ** 3 * d - 18 * np.sqrt(3) * b_c * d ** 2 * (a + 11 * d) + b_c ** 2 * d * (4 * a + 135 * d) + d ** 2 * (a ** 2 + 54 * a * d + 297 * d ** 2)) * 1 ** 9 - 41472 * (np.sqrt(3) * b_c - 6 * d) * d ** 3 * (-2 * b_c ** 2 + 11 * np.sqrt(3) * b_c * d - 3 * d * (a + 11 * d)) * 1 ** 10 + (746496 * d ** 4 * (5 * b_c ** 2 - 22 * np.sqrt(3) * b_c * d + 2 * d * (a + 33 * d)) * 1 ** 11)/11 + 248832 * (np.sqrt(3) * b_c - 6 * d) * d ** 5 * 1 ** 12 + (2985984 * d ** 6 * 1 ** 13)/13
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, _ = _generate_higher_order_dist_funcs2(a, b_c, d, ys * norm)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    return _bisection_solver(f=cdfs, x_dim=x_dim, n=n)


def _higher_order_sample_test(matrices, n=10, name='quadratic', seed=None):
    a = matrices[:, 0, 0]
    b_c = matrices[:, 0, 1] + matrices[:, 1, 0]
    d = matrices[:, 1, 1]
    norm = ((a - np.sqrt(3) * b_c + 3 * d)**3
            + 3 * (np.sqrt(3) * b_c - 6 * d) * (a - np.sqrt(3) * b_c + 3 * d)**2
            + 12 * (-(np.sqrt(3) * b_c**3) + a**2 * d + 18 * b_c**2 * d - 30 * np.sqrt(3) * b_c * d**2 + 45 * d**3 + a * (b_c**2 - 6 * np.sqrt(3) * b_c * d + 18 * d**2))
            - 6 * (np.sqrt(3) * b_c - 6 * d) * (-b_c**2 + 10 * np.sqrt(3) * b_c * d - 6 * d * (a + 5 * d))
            + (432 * d * (b_c**2 - 5 * np.sqrt(3) * b_c * d + d * (a + 15 * d))) / 5
            + 144 * (np.sqrt(3) * b_c - 6 * d) * d**2
            + (1728 * d**3) / 7)
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, pdfs = _generate_higher_order_dist_funcs(a, b_c, d, ys * norm)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    xs = _bisection_solver(f=cdfs, x_dim=x_dim, n=10)
    return ys, xs, pdfs(xs)


def _quad_sample_test(matrices, name='quadratic', seed=None):
    a = matrices[:, 0, 0]
    b_c = matrices[:, 0, 1] + matrices[:, 1, 0]
    d = matrices[:, 1, 1]
    norm = a + 2 * np.sqrt(3) * b_c + 13.5 * d
    _a = a / norm
    _b_c = b_c / norm
    _d = d / norm
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, pdfs = _generate_dist_funcs(_a, _b_c, _d, ys)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    print(x_dim.shape)
    print(a.shape)
    x_dim = tf.Print(x_dim, [x_dim])
    xs = _bisection_solver(f=cdfs, x_dim=x_dim, n=10)
    return ys, xs, pdfs(xs)


def _generate_dist_funcs(a, b_c, d, offsets):
    def f(x):
        return (a * x + np.sqrt(3) * x**2 * (b_c)
                - np.sqrt(3) * x * b_c
                + d * 4 * x**3
                - d * 6 * x**2
                + 3 * d * x - offsets)

    def df(x):
        return (a + b_c * np.sqrt(3) * (2 * x - 1)
                + 3 * d * (2 * x - 1)**2)

    return f, df


def _generate_higher_order_dist_funcs(a, b_c, d, offsets):
    def f(x):
        return ((a - np.sqrt(3) * b_c + 3 * d)**3 * x
                + 3 * (np.sqrt(3) * b_c - 6 * d) * (a - np.sqrt(3) * b_c + 3 * d)**2 * x**2
                + 12 * (-(np.sqrt(3) * b_c**3) + a**2 * d + 18 * b_c**2 * d - 30 * np.sqrt(3) * b_c * d**2 + 45 * d**3 + a * (b_c**2 - 6 * np.sqrt(3) * b_c * d + 18 * d**2)) * x**3
                - 6 * (np.sqrt(3) * b_c - 6 * d) * (-b_c**2 + 10 * np.sqrt(3) * b_c * d - 6 * d * (a + 5 * d)) * x**4
                + (432 * d * (b_c**2 - 5 * np.sqrt(3) * b_c * d + d * (a + 15 * d)) * x**5) / 5
                + 144 * (np.sqrt(3) * b_c - 6 * d) * d**2 * x**6
                + (1728 * d**3 * x**7) / 7 - offsets)

    def df(x):
        return (a + b_c * np.sqrt(3) * (2 * x - 1)
                + 3 * d * (2 * x - 1)**2)**3

    return f, df

def _generate_higher_order_dist_funcs2(a, b_c, d, offsets):
    def f(x):
        return ((a ** 6 - 6 * a ** 5 * (np.sqrt(3) * b_c - 3 * d) + 45 * a ** 4 * (b_c ** 2 - 2 * np.sqrt(3) * b_c * d + 3 * d ** 2) - 60 * a ** 3 * (np.sqrt(3) * b_c ** 3 - 9 * b_c ** 2 * d + 9 * np.sqrt(3) * b_c * d ** 2 - 9 * d ** 3) + 135 * a ** 2 * (b_c ** 4 - 4 * np.sqrt(3) * b_c ** 3 * d + 18 * b_c ** 2 * d ** 2 - 12 * np.sqrt(3) * b_c * d ** 3 + 9 * d ** 4) - 54 * a * (np.sqrt(3) * b_c ** 5 - 15 * b_c ** 4 * d + 30 * np.sqrt(3) * b_c ** 3 * d ** 2 - 90 * b_c ** 2 * d ** 3 + 45 * np.sqrt(3) * b_c * d ** 4 - 27 * d ** 5) + 27 * (b_c ** 6 - 6 * np.sqrt(3) * b_c ** 5 * d + 45 * b_c ** 4 * d ** 2 - 60 * np.sqrt(3) * b_c ** 3 * d ** 3 + 135 * b_c ** 2 * d ** 4 - 54 * np.sqrt(3) * b_c * d ** 5 + 27 * d ** 6)) * x - 6 * (np.sqrt(3) * b_c - 6 * d) * (-a + np.sqrt(3) * b_c - 3 * d) ** 5 * x ** 2 + 12 * (a - np.sqrt(3) * b_c + 3 * d) ** 4 * (5 * b_c ** 2 - 22 * np.sqrt(3) * b_c * d + 2 * d * (a + 33 * d)) * x ** 3 + 60 * (np.sqrt(3) * b_c - 6 * d) * (-a + np.sqrt(3) * b_c - 3 * d) ** 3 * (-2 * b_c ** 2 + 11 * np.sqrt(3) * b_c * d - 3 * d * (a + 11 * d)) * x ** 4 + 432 * (a - np.sqrt(3) * b_c + 3 * d) ** 2 * (b_c ** 4 - 12 * np.sqrt(3) * b_c ** 3 * d - 18 * np.sqrt(3) * b_c * d ** 2 * (a + 11 * d) + b_c ** 2 * d * (4 * a + 135 * d) + d ** 2 * (a ** 2 + 54 * a * d + 297 * d ** 2)) * x ** 5 - 288 * (np.sqrt(3) * b_c - 6 * d) * (np.sqrt(3) * b_c ** 5 - 57 * b_c ** 4 * d - 10 * a ** 3 * d ** 2 + 306 * np.sqrt(3) * b_c ** 3 * d ** 2 - 1944 * b_c ** 2 * d ** 3 + 1782 * np.sqrt(3) * b_c * d ** 4 - 1782 * d ** 5 - 10 * a ** 2 * d * (b_c ** 2 - 7 * np.sqrt(3) * b_c * d + 21 * d ** 2) - a * (b_c ** 4 - 28 * np.sqrt(3) * b_c ** 3 * d + 462 * b_c ** 2 * d ** 2 - 756 * np.sqrt(3) * b_c * d ** 3 + 1134 * d ** 4)) * x ** 6 + (1728 * (b_c ** 6 - 42 * np.sqrt(3) * b_c ** 5 * d - 420 * np.sqrt(3) * b_c ** 3 * d ** 2 * (a + 12 * d) + 30 * b_c ** 4 * d * (a + 42 * d) - 84 * np.sqrt(3) * b_c * d ** 3 * (5 * a ** 2 + 90 * a * d + 297 * d ** 2) + 90 * b_c ** 2 * d ** 2 * (a ** 2 + 56 * a * d + 315 * d ** 2) + 4 * d ** 3 * (5 * a ** 3 + 315 * a ** 2 * d + 2835 * a * d ** 2 + 6237 * d ** 3)) * x ** 7)/7 - 2592 * (np.sqrt(3) * b_c - 6 * d) * d * (-b_c ** 4 + 18 * np.sqrt(3) * b_c ** 3 * d + 12 * np.sqrt(3) * b_c * d ** 2 * (5 * a + 33 * d) - 2 * b_c ** 2 * d * (5 * a + 126 * d) - 2 * d ** 2 * (5 * a ** 2 + 90 * a * d + 297 * d ** 2)) * x ** 8 + 34560 * d ** 2 * (b_c ** 4 - 12 * np.sqrt(3) * b_c ** 3 * d - 18 * np.sqrt(3) * b_c * d ** 2 * (a + 11 * d) + b_c ** 2 * d * (4 * a + 135 * d) + d ** 2 * (a ** 2 + 54 * a * d + 297 * d ** 2)) * x ** 9 - 41472 * (np.sqrt(3) * b_c - 6 * d) * d ** 3 * (-2 * b_c ** 2 + 11 * np.sqrt(3) * b_c * d - 3 * d * (a + 11 * d)) * x ** 10 + (746496 * d ** 4 * (5 * b_c ** 2 - 22 * np.sqrt(3) * b_c * d + 2 * d * (a + 33 * d)) * x ** 11)/11 + 248832 * (np.sqrt(3) * b_c - 6 * d) * d ** 5 * x ** 12 + (2985984 * d ** 6 * x ** 13)/13
                - offsets)
    def df(x):
        return (a + b_c * np.sqrt(3) * (2 * x - 1)
                + 3 * d * (2 * x - 1)**2)**6
    return f, df

if __name__ == '__main__':

    # a_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    # b_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    # a_nparray = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]).astype(np.float32)
    # b_nparray = a_nparray
    # c_nparray = b_nparray
    # d_nparray = a_nparray

    # a = tf.constant(a_nparray)
    # b = tf.constant(b_nparray)

    arrays = [np.array([[0.0333175, -0.0023912771], [-0.0023912771, 0.00089279143]]).astype(np.float32) for _ in range(100)]
    mat = np.stack(arrays)
    matrices = tf.constant(mat)

    cdfs, samples, cdfs_inv = _higher_order_sample_alt_test(matrices)

    with tf.Session() as sess:
        tf.global_variables_initializer()
        samples, cdfs, cdfs_inv = sess.run([samples, cdfs, cdfs_inv])
        print(samples)
        print(cdfs)
        print(cdfs_inv)
        print(cdfs - cdfs_inv)
        print(samples[samples > 1].shape)
        print(cdfs[cdfs < 0.5].shape)
        plt.plot(samples, cdfs, 'k.')
        plt.plot(samples, cdfs_inv, 'b.')
        plt.show()


