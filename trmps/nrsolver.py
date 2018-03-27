import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def _nr_solver(f, df, x_dim, n=10):
    # solves in boundary [0,1]
    # n is the number of iterations
    def _nr_solver_helper(counter, x):
        return counter + 1, tf.floormod(x - (f(x) / df(x)), 1.0)

    cond = lambda c, *args: tf.less(c, n)
    x_initial = tf.zeros(x_dim) + 1.0
    x_initial = tf.Print(x_initial, [x_dim])
    _, _x = tf.while_loop(cond=cond,
                          body=_nr_solver_helper,
                          loop_vars=[0, x_initial],
                          # shape_invariants=(tf.TensorShape([]),
                          #                  tf.TensorShape([])),
                          parallel_iterations=1)
    return _x


def _test_f(x):
    return (x + 3) * (x - 5) * (x - .25)


def _test_df(x):
    return 3.0 * (x ** 2 - 1.5 * x - 4.83333333)


def _generate_dist_funcs(a, b, offsets):
    def f(x):
        return a * x + b * (3 * x - 6 * x**2 + 4 * x**3) - offsets

    def df(x):
        return a + 3 * b * (2 * x - 1)**2
    return f, df


if __name__ == '__main__':

    result = _nr_solver(_test_f, _test_df, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer()
        computed_result = sess.run([result])
        print(computed_result)
