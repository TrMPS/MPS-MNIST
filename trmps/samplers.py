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



if __name__ == '__main__':

    # a_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    # b_nparray = np.random.uniform(size=100, low=-10, high=10).astype(np.float32)
    a_nparray = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ]).astype(np.float32)
    b_nparray = a_nparray

    a = tf.constant(a_nparray)
    b = tf.constant(b_nparray)

    cdfs, samples, cdfs_inv = _quad_sample_test(a, b)

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


