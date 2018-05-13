import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _bisection_solver(f, x_dim, n=9):
    # solves in boundary [0,1]
    # n is the number of iterations
    def _bisection_solver_helper(counter, a, b):
        midpoint = (a + b) / 2
        f_mid = f(midpoint)
        # f_mid = tf.Print(f_mid, [f_mid, midpoint], summarize=10)

        # def _new_b_finder(_f_mid):
        #     cond = tf.less(_f_mid, 0.0)
        #     return tf.cast(cond, tf.float32)
        # mid_mask = tf.map_fn(_new_b_finder, f_mid)
        mid_mask = tf.cast(tf.less(f_mid, 0.0), tf.float32)
        mid_mask_inverted = 1.0 - mid_mask
        new_b_1 = mid_mask_inverted * midpoint
        new_b_2 = mid_mask * b
        new_b = new_b_1 + new_b_2
        # new_b = tf.Print(new_b, [mid_mask, new_b, b], summarize=10)
        new_a_1 = mid_mask_inverted * a
        new_a_2 = mid_mask * midpoint
        new_a = new_a_1 + new_a_2
        return counter + 1, new_a, new_b

    cond = lambda c, *args: tf.less(c, n)
    a_initial = tf.zeros(x_dim)
    b_initial = tf.zeros(x_dim) + 1.0
    _, a, b = tf.while_loop(cond=cond,
                          body=_bisection_solver_helper,
                          loop_vars=[0, a_initial, b_initial],
                          # shape_invariants=(tf.TensorShape([]),
                          #                  tf.TensorShape([])),
                          parallel_iterations=1)
    return a

def _test_f(x):
    return (x + 3) * (x - 5) * (x - .6)


def _test_df(x):
    return 3.0 * (x ** 2 - 1.5 * x - 4.83333333)


if __name__ == '__main__':
    result = bisection_solver(_test_f, 1, n=10)
    # This test doesn't currently work because of the way the test function starts positive and goes negative, but this isn't a problem for the intended use case of the bisection_solver
    with tf.Session() as sess:
        tf.global_variables_initializer()
        computed_result = sess.run([result])
        print(computed_result)
