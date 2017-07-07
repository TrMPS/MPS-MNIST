import tensorflow as tf 
import numpy as np
from functools import reduce

shape = [2, 1, 1, 4, 4]
size = reduce((lambda x, y: x*y), shape)
B = np.arange(size).reshape(shape)
floor = 3
cap = 5


bond = tf.Variable(B, dtype=tf.float32)    
max_size = tf.Variable(cap, dtype=tf.int32)
min_size = tf.Variable(floor, dtype=tf.int32)
_threshold = tf.Variable(10 ** (-4), dtype=tf.float32)


bond_reshaped = tf.transpose(bond, perm=[1, 3, 0, 2, 4])
dims = tf.shape(bond_reshaped)
l_dim = dims[0] * dims[1]
r_dim = dims[2] * dims[3] * dims[4]
bond_flattened = tf.reshape(bond_reshaped, [l_dim, r_dim])
s, u, v = tf.svd(bond_flattened)

filtered_s = tf.boolean_mask(s, tf.greater(s, _threshold))
s_size = tf.size(filtered_s)

case1 = lambda : min_size 
case2 = lambda : max_size
case3 = lambda : s_size
m = tf.case({tf.less(s_size, min_size): case1, tf.greater(s_size, max_size): case2}, default=case3, exclusive=True)

# make s into a matrix
s_mat = tf.diag(s[0:m])

# make u, v into suitable matrices
u_cropped = u[:, 0:m] 
v_cropped = tf.transpose(v[:, 0:m])

# make a_ 
a_prime_j = tf.reshape(u_cropped, [dims[0], dims[1], m])
sv = tf.matmul(s_mat, v_cropped)
a_prime_j1_mixed = tf.reshape(sv, [m, dims[2], dims[3], dims[4]])
a_prime_j1 = tf.transpose(a_prime_j1_mixed, perm=[1, 2, 0, 3])

new_bond = tf.einsum('mij,lnjk->lmnik', a_prime_j, a_prime_j1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(bond.eval())
    print(bond_flattened.eval())
    print(new_bond.eval())
