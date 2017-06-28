import tensorflow as tf 
import numpy as np

'''
1. Store the C matrix computation cumulatively
'''

class MPS(object):

	def __init__(self, d_matrix, d_feature, d_output, batch_size, input_size, rate_of_change=0.1):
		# structure parameters
		self.input_size = input_size
		self.d_matrix = d_matrix
		self.d_feature = d_feature
		self.d_output = d_output

		# training parameters
		self.batch_size = batch_size
		self.rate_of_change = rate_of_change

		# Initialise the nodes, input and output
		self.nodes = self.make_nodes()
		self.phi = tf.placeholder(tf.float32, shape=[input_size, batch_size, d_feature])
		self.delta = tf.placeholder(tf.float32, shape=[batch_size, d_output])

	def prepare(self):
		# Calculate the bond matrix
		self.bond = tf.einsum('lmij,njk->lmnik', self.nodes[1], self.nodes[2]) # ik: matrix leg, mn: input leg, l: output leg

		# Calculate the first C, where f_Nl = B_tl * C_ikmn
		C_1 = tf.einsum('ni,tn->ti', self.nodes[0], self.phi[0]) # left part
		C_2 = tf.einsum('mij,tm->tij', self.nodes[3], self.phi[3]) # need to check whether the input size is large enough!
		for i in range(4, self.input_size-1):
			C_2 = tf.einsum('tij,mjk,tm->tik', C_2, self.nodes[i], self.phi[i])
		C_2 = tf.einsum('tij,mj,tm->ti', C_2, self.nodes[-1], self.phi[-1])

		self.C = tf.einsum('ti,tk,tm,tn->tmnik', C_1, C_2, self.phi[1], self.phi[2])


	def batch_train(self, phi, delta):
		# Update the nodes and move along the string
		for i in range(1, self.input_size-3):

			self.update_bond()
			self.update_nodes(i)
			self.move_one_step(i)

		# Update the last bond
		self.update_bond()
		self.update_nodes(self.input_size-3)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(sess.run(self.nodes, {self.phi: phi, self.delta: delta}))


	# Test function
	def test(self, phi, delta):
		f = tf.einsum('lmnik,tmnik->tl', self.bond, self.C)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(sess.run(f, {self.phi: phi, self.delta: delta}))
			writer = tf.summary.FileWriter("output", sess.graph)
			writer.close()



	# =====================
	# Helper functions
	# =====================

	def make_random_normal(self, shape, mean=0, stddev=1):
		return tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev))


	def make_nodes(self):
		nodes = []

		# First node
		nodes.append(self.make_random_normal([self.d_feature, self.d_matrix]))
		# The second node with output leg attached
		nodes.append(self.make_random_normal([self.d_output, self.d_feature, self.d_matrix, self.d_matrix]))
		# The rest of the matrix nodes
		for i in range(self.input_size-2):
			nodes.append(self.make_random_normal([self.d_feature, self.d_matrix, self.d_matrix]))
		# Last node
		nodes.append(self.make_random_normal([self.d_feature, self.d_matrix]))

		return nodes

	def update_bond(self):
		'''
		Update the bond matrix with gradient descent
		'''
		f = tf.einsum('lmnik,tmnik->tl', self.bond, self.C)
		gradient = tf.einsum('tl,tmnik->lmnik', self.delta - f, self.C)
		delta_bond = self.rate_of_change * gradient
		self.bond += delta_bond


	def update_nodes(self, index):
		'''
		Update the relevant notes
		'''
		a_prime_j, a_prime_j1 = self.bond_decomposition(self.d_matrix)
		self.nodes[index] = a_prime_j
		self.nodes[index+1] = a_prime_j1

	def move_one_step(self, index):
		'''
		Move one step along the chain
		We achieve this by updating the bond and C matrices
		'''
		a_inverse_j = tf.matrix_inverse(self.nodes[index])
		a_inverse_j2 = tf.matrix_inverse(self.nodes[index+2])

		self.bond = tf.einsum('lmnik,mij,okh->lnojh', self.bond, a_inverse_j, self.nodes[i+2])
		self.C = tf.einsum('tmnik,mij,okh->tnojh', self.C, self.nodes[i], a_inverse_j2)

	def bond_decomposition(self, m):
		"""
		:param bond:
		:param m:
		:return:
		"""
		s, a_prime_j, v = tf.svd(self.bond)
		filtered_s = self.highest_values(s, m)
		a_prime_j1 = tf.matmul(filtered_s, v)
		return (a_prime_j, a_prime_j1)


	def highest_values(self, matrix, m):
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


if __name__ == '__main__':

	# Model parameters
	input_size = 10
	d_feature = 2
	d_matrix = 5
	d_output = 3
	rate_of_change = 0.2
	batch_size = 2

	# Make up input and output
	phi = np.random.normal(size=(input_size, batch_size, d_feature))
	delta = [[1, 0, 0], [0, 1, 0]]

	# Initialise the model
	network = MPS(d_matrix, d_feature, d_output, batch_size, input_size,
					rate_of_change=rate_of_change)
	network.prepare()
	network.test(phi, delta)
