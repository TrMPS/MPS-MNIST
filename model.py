import tensorflow as tf 
import numpy as np

'''
TODO: 
1. Make sure that input_size is big enough for the calculation to be meaningful
'''

# Model parameters
input_size = 10
d_feature = 2 
d_matrix = 5
d_output = 3 

# Initialise the nodes and the inputs
phi = [tf.Variable([1.0, 0], dtype=tf.float32)] * input_size # input, Phi_s, shape = d_feature 
nodes = [] # MPS, A^(alpha, beta)_s

# A function for generating nodes
def make_random_normal(shape, mean=0, stddev=1):
	return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))

# First node
nodes.append(make_random_normal([d_matrix, d_feature])) 

# The second node with output leg attached 
nodes.append(make_random_normal([d_matrix, d_matrix, d_feature, d_output]))

# The rest of the matrix nodes 
for i in range(input_size-2):
	nodes.append(make_random_normal([d_matrix, d_matrix, d_feature])) 

# Last node
nodes.append(make_random_normal([d_matrix, d_feature])) 


# Calculate the first B 
B = tf.einsum('ijml,jkn->ikmnl', nodes[1], nodes[2]) # ik: matrix leg, mn: input leg, l: output leg

# Calculate the first C, where f_l = B_ikmnl * C_ikmn
C_1 = tf.einsum('in,n->i', nodes[0], phi[0]) # left part


C_2 = tf.einsum('ijm,m->ij', nodes[3], phi[3]) # need to check whether the input size is large enough! 
for i in range(4, input_size-1):
	C_2 = tf.einsum('ij,jkm,m->ik', C_2, nodes[i], phi[i])
C_2 = tf.einsum('ij,jm,m->i', C_2, nodes[-1], phi[-1])

C = tf.einsum('i,k,m,n->ikmn', C_1, C_2, phi[1], phi[2])


# Calculate the decision function for one data point
f = tf.einsum('ikmnl,ikmn->l', B, C)


# Try 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(f))
	

def bond_decomposition(bond, m, session = None):
	"""
	:param bond:
	:param m:
	:return:
	"""
	rate_of_change = 5.0
	bond_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, None])
	delta_bond = tf.placeholder(tf.float32, shape=[None, None, None, None])
	bond_change = tf.mul(delta_bond, rate_of_change)
	bond_prime_op = tf.add(bond_placeholder, bond_change)
	if session is None:
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			bond_prime = sess.run(bond_prime_op, {bond_placeholder: bond})
	else:
		sess.run(tf.global_variables_initializer())
		bond_prime = sess.run(bond_prime_op, {bond_placeholder: bond})
	s, a_prime_j, v = tf.svd(bond_prime)
	filtered_s = highest_values(s, m)
	a_prime_j1 = filtered_s * v
	return (a_prime_j, a_prime_j1)


def highest_values(matrix, m):
	array_np = np.array(matrix)
	flattened = np.ravel(array_np)
	highest_vals = np.unique(np.sort(array_np.flatten())[-m:])
	output = np.zeros(array_np.shape)
	for val in highest_vals:
		masked_array = ma.masked_where(array_np != val, array_np)
		filled = masked_array.filled(0.0)
		output += filled
	return (output)



