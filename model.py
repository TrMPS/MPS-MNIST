import tensorflow as tf 
import numpy as np

'''
TODO: 
1. Make sure that input_size is big enough for the calculation to be meaningful
2. Change "Variable" to placeholder for phi and delta 
3. Rewrite phi to read in different input 
'''

# A function for generating nodes
def make_random_normal(shape, mean=0, stddev=1):
	return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))


def make_nodes(d_matrix, d_feature, d_output):
	nodes = []

	# First node
	nodes.append(make_random_normal([d_feature, d_matrix])) 
	# The second node with output leg attached 
	nodes.append(make_random_normal([d_output, d_feature, d_matrix, d_matrix]))
	# The rest of the matrix nodes 
	for i in range(input_size-2):
		nodes.append(make_random_normal([d_feature, d_matrix, d_matrix])) 	
	# Last node
	nodes.append(make_random_normal([d_feature, d_matrix])) 

	return nodes


def bond_decomposition(bond, m):
	"""
	:param bond:
	:param m:
	:return:
	"""
	s, a_prime_j, v = tf.svd(bond)
	filtered_s = highest_values(s, m)
	a_prime_j1 = tf.matmul(filtered_s, v)
	return (a_prime_j, a_prime_j1)


def highest_values(matrix, m):
	return matrix

# 	array_np = np.array(matrix)
# 	flattened = np.ravel(array_np)
# 	highest_vals = np.unique(np.sort(array_np.flatten())[-m:])
# 	output = np.zeros(array_np.shape)
# 	for val in highest_vals:
# 		masked_array = ma.masked_where(array_np != val, array_np)
# 		filled = masked_array.filled(0.0)
# 		output += filled
# 	return (output)

def bond_prime(bond, delta, C, rate_of_change): 
	f = tf.einsum('lmnik,tmnik->tl', bond, C)
	gradient = tf.einsum('tl,tmnik->lmnik', delta - f, C)
	delta_bond = rate_of_change * gradient 
	return bond + delta_bond


def update_nodes(bond, index, nodes):
	assert index < length(nodes)
	a_prime_j, a_prime_j1 = bond_decomposition(bond, d_matrix)
	nodes[index] = a_prime_j
	nodes[index+1] = a_prime_j1

def move_one_step(bond, C, index, nodes):
	a_inverse_j = tf.matrix_inverse(nodes[index])
	a_inverse_j2 = tf.matrix_inverse(nodes[index+2])

	bond = tf.einsum('lmnik,mij,okh->lnojh', bond, a_inverse_j, nodes[i+2])
	C = tf.einsum('tmnik,mij,okh->tnojh', C, nodes[i], a_inverse_j2)

	return bond, C 


if __name__ == '__main__':

	# Model parameters
	input_size = 10
	d_feature = 2 
	d_matrix = 5
	d_output = 3 
	rate_of_change = 5.0
	batch_size = 2

	# Make up input and output 
	feature = [[1, 0], [0, 1]]
	output = [[1, 0, 0], [0, 1, 0]]
	
	
	# Initialise the nodes, the inputs and the outputs
	phi = [tf.Variable(feature, dtype=tf.float32)] * input_size # input, Phi_s, shape = d_feature 
	delta = tf.Variable(output, dtype=tf.float32)
	nodes = make_nodes(d_matrix, d_feature, d_output)
		
	
	# Calculate the first B 
	bond = tf.einsum('lmij,njk->lmnik', nodes[1], nodes[2]) # ik: matrix leg, mn: input leg, l: output leg
	
	# Calculate the first C, where f_Nl = B_tl * C_ikmn
	C_1 = tf.einsum('ni,tn->ti', nodes[0], phi[0]) # left part
	
	
	C_2 = tf.einsum('mij,tm->tij', nodes[3], phi[3]) # need to check whether the input size is large enough! 
	for i in range(4, input_size-1):
		C_2 = tf.einsum('tij,mjk,tm->tik', C_2, nodes[i], phi[i])
	C_2 = tf.einsum('tij,mj,tm->ti', C_2, nodes[-1], phi[-1])
	
	C = tf.einsum('ti,tk,tm,tn->tmnik', C_1, C_2, phi[1], phi[2])
	
	
	# Loop through and update 
	for i in range(1, input_size-3):

		bond = bond_prime(bond, delta, C, rate_of_change)
		update_nodes(bond, i, nodes)
		bond, C = move_one_step(bond, C, i, nodes)

	bond = bond_prime(bond, delta, C)
	update_nodes(bond, input_size-3, nodes)


	# Try 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(bond))



	




