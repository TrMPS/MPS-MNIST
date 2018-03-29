import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from solvers import _bisection_solver, _generate_dist_funcs

def quad_sample(a, b, n, name='quadratic', seed=None):
    norm = a + b
    _a = a / norm
    _b = b / norm
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, _ = _generate_dist_funcs(_a, _b, ys)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    return _bisection_solver(cdfs, x_dim, n=n)

def _quad_sample_test(a, b, name='quadratic', seed=None):
    norm = a + b
    _a = a / norm
    _b = b / norm
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, pdfs = _generate_dist_funcs(_a, _b, ys)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    print(x_dim.shape)
    print(a.shape)
    x_dim = tf.Print(x_dim, [x_dim])
    xs = _bisection_solver(f=cdfs, x_dim=x_dim, n=10)
    return ys, xs, pdfs(xs)


def _higher_order_sample_test(a, b, n=10, name='higher order sample', seed=None):
    norm = a**3 + 3 * a**2 * b + (27 / 5) * (a * b**2) + (27 / 7) * (b**3)
    _a = a / norm
    _b = b / norm
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, pdfs = _generate_higher_order_dist_funcs(_a, _b, ys)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    print(x_dim.shape)
    print(a.shape)
    x_dim = tf.Print(x_dim, [x_dim])
    xs = _bisection_solver(f=cdfs, x_dim=x_dim, n=n)
    return ys, xs, pdfs(xs)

def higher_order_sample(a, b, n, name='higher order sample', seed=None):
    norm = a**3 + 3 * a**2 * b + (27 / 5) * (a * b**2) + (27 / 7) * (b**3)
    _a = a / norm
    _b = b / norm
    ys = tf.random_uniform([tf.shape(a)[0]], minval=0, maxval=1, seed=seed)
    cdfs, pdfs = _generate_higher_order_dist_funcs(_a, _b, ys)
    x_dim = tf.reshape(tf.shape(a)[0], [1])
    return _bisection_solver(f=cdfs, x_dim=x_dim, n=n)



def _generate_higher_order_dist_funcs(a, b, offsets):
    def f(x):
        return ((12 * b * x**3) * (a**2 + 18 * a * b + 45 * b**2)
         + (432 / 5) * (b**2 * x**5) * (a + 15 * b)
         - (216 * b**2 * x**4) * (a + 5 * b)
         - 18 * b * x**2 * (a + 3 * b)**2
         + x * (a + 3 * b)**3
         + (1728 / 7) * b**3 * x**7
         - 864 * b**3 * x**6 - offsets)

    def df(x):
        return (a + 3 * b * (2 * x - 1)**2) ** 3
    return f, df


if __name__ == '__main__':

    # a_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    # b_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    a_nparray = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]).astype(np.float32)
    b_nparray = a_nparray

    a = tf.constant(a_nparray)
    b = tf.constant(b_nparray)

    cdfs, samples, cdfs_inv = _higher_order_sample_test(a, b)

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


