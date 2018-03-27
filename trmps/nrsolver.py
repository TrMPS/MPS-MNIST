import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def _nr_solver(f, df, n=10):
    # solves in boundary [0,1]
    # n is the number of iterations
    def _nr_solver_helper(counter, x):
        return counter + 1, tf.floormod(x - (f(x) / df(x)), 1.0)


    cond = lambda c, *args: tf.less(c, n)
    _, _x = tf.while_loop(cond=cond,
                          body=_nr_solver_helper,
                          loop_vars=[0, 0.0],
                          # shape_invariants=(tf.TensorShape([]),
                          #                  tf.TensorShape([])),
                          parallel_iterations=1)
    return _x

def _test_f(x):
    return (x+3) * (x-5) * (x-.25)

def _test_df(x):
    return 3.0*(x**2 - 1.5*x -4.83333333)

if __name__ == '__main__':

    result = _nr_solver(_test_f, _test_df)

    with tf.Session() as sess:
        tf.global_variables_initializer()
        computed_result = sess.run([result])
        print(computed_result)
