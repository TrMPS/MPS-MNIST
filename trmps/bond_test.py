import tensorflow as tf
import numpy as np
import utils

min_singular_value = 10 
max_size = 20 

bond = tf.constant()
def _bond_decomposition(self, bond, max_size, min_size=3):
    """

    :param self:
    :param bond:
    :param max_size:
    :param min_size:
    :param min_singular_value:
    :return:
    """
    """
    Decomposes bond, so that the next step can be done.
    :param bond:
    :param m:
    :return:
    """
    with tf.name_scope("bond_decomposition"):
        bond_reshaped = tf.transpose(bond, perm=[1, 3, 0, 2, 4])
        # bond_reshaped = tf.Print(bond_reshaped, [tf.shape(bond_reshaped), tf.shape(bond)], summarize = 1000, message = "bond reshaped, bond")

        dims = tf.shape(bond_reshaped)
        l_dim = dims[0] * dims[1]
        r_dim = dims[2] * dims[3] * dims[4]
        bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
        s, u, v = tf.svd(bond_flattened)
        filtered_u = utils.check_nan(u, 'u', replace_nan=True)
        filtered_v = utils.check_nan(v, 'v', replace_nan=True)

        filtered_s = tf.boolean_mask(s, tf.greater(s, min_singular_value))
        s_size = tf.size(filtered_s)
        # s_size = tf.Print(s_size, [s_size], message='bond dim: ')

        case1 = lambda: min_size
        case2 = lambda: max_size
        case3 = lambda: s_size
        m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3,
                    exclusive=True)
        # if self.verbose != 0:
        #     m = tf.Print(m, [m, s[m-5:m]], first_n=self.verbose, summarize=5, message='bond: ')

        # make s into a matrix
        s_mat = tf.diag(s[0:m])

        # make u, v into suitable matrices
        u_cropped = filtered_u[:, 0:m]
        v_cropped = tf.transpose(filtered_v[:, 0:m])

        # make a_
        a_prime_j = tf.reshape(u_cropped, [dims[0], dims[1], m])

        sv = tf.matmul(s_mat, v_cropped)
        a_prime_j1 = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
        # a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])
        # will do this in the update_right/update_left functions from now on as else transpose twice for udpate_left

    return (a_prime_j, a_prime_j1)

def _svd_decomp(self, bond_flattened, max_size):
    s, u, v = tf.svd(bond_flattened)
    filtered_u = utils.check_nan(u, 'u', replace_nan=True)
    filtered_v = utils.check_nan(v, 'v', replace_nan=True)
    filtered_s = tf.boolean_mask(s, tf.greater(s, self.min_singular_value))
    s_size = tf.size(filtered_s)

    m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3,
                exclusive=True)
    s_mat = tf.diag(s[0:m])
    v_cropped = filtered_v[:, 0:m]
    A = filtered_u[:, 0:m]
    B = tf.matmul(s_mat, v_cropped, transpose_b=True)

    if s_size > max_size:

        # update B when A is in canonical form 
        def iter_step(c, A, B, tol):
            B = tf.matmul(A, bond_flattened, transpose_a=True)

            s, u, v = tf.svd(B)
            A = tf.matmul(A, u)
            s = tf.expand_dims(s, 0)
            A = tf.multiply(A, s)

            A = tf.matmul(bond_flattened, v)
            tol = 1 - tf.tensordot(A, A, [[0, 1], [0, 1]])
            s, u_, v_ = tf.svd(A)
            B = tf.matmul(v_, v, transpose_a=True, transpose_b=True)
            s_reshaped = tf.expand_dims(s, 1)
            B = tf.multiply(s_reshaped, B)
            A = u_

            return c+1, A, B, tol


        cond = lambda c, A, B, t: tf.logical_and(tf.less(c, 10), tf.more(t, tolerance))

        tf.while_loop(cond=cond, body=iter_step, loop_vars=[0, u_]
                                                        shape_invariants=[]
                                                        parallel_iterations=5, name="rightSweep")



