import numpy as np 	
import tensorflow as tf


class Quadratic(object):
	"""
	A 'quadratic' distribution where pdf(x) = (a + \sqrt(2x - 1) b)^2 

	Example usage:

	from distribution import Quadratic 
	import numpy as np 
	import tensorflow as tf 

	# Create a and b arrays to generate from 
	a_nparray = np.random.uniform(size=10000, low=-10, high=10).astype(np.float32)
	b_nparray = np.random.uniform(size=10000, low=-10, high=10).astype(np.float32)

	a = tf.constant(a_nparray)
	b = tf.constant(b_nparray)

	# Generate from the distribution and calculate the cdf of the generated samples 
	quad_dist = Quadratic(a, b)
	samples = quad_dist.sample() 
	cdfs = quad_dist.cdf(samples)

	# Run a tensorflow session 
	with tf.Session() as sess:
		tf.global_variables_initializer()
		samples, cdfs = sess.run([samples, cdfs])
		print(samples[samples > 1].shape)
		print(cdfs[cdfs < 0.5].shape)
	"""

	def __init__(self, 
				 a, 
				 b, 
				 tol=1e-6,
				 name="quadratic"):
		"""
		Initialise the model with arrays of a and b 

		:param a: tf.Tensor
			1D Tensor 
		:param b: tf.Tensor
			1D Tensor 
		:param tol: float 
			if abs(b) is smaller than tol, treat it as zero when generating
		"""

		norm = tf.sqrt(a * a + b * b) 
		self._a = a/norm
		self._b = b/norm
		self._a_sq = tf.square(self._a)
		self._b_sq = tf.square(self._b)
		self._tol = tol
		self._n_params = tf.shape(a)[0] 

	def sample(self, seed=None):
		"""
		Generate one sample from each pair of (a, b)
		"""
		cdfs = tf.random_uniform([self._n_params], minval=0, maxval=1, seed=seed)
		return self._inverse_cdf(cdfs)

	def prob(self, x):
		"""
		Calculate the probabilities of an array of x following the 
		pdfs specified by the a and b arrays 

		Note length(x) must be equal to length(b)
		"""
		u = 2 * x - 1
		middle_term = 2 * np.sqrt(3) * u * self._a * self._b 
		last_term = 3 * tf.square(u) * self._b_sq 
		return (self._a_sq + middle_term + last_term)

	def cdf(self, x):
		"""
		Calculate the cdfs of an array of x 
		"""
		first_term = self._a_sq * x 
		second_term = 2 * np.sqrt(3) * self._a * self._b * (x - 1) * x 
		third_term =  self._b_sq * ((2 * x - 1) ** 3 + 1)/2 
		return (first_term + second_term + third_term)

	def _inverse_cdf_zero_b(self, f, a):
		return f/tf.square(a)

	def _inverse_cdf_finite_b(self, f, a, b):
		c = a - np.sqrt(3) * b
		to_cube_root = np.sqrt(3) * c ** 3 / b + 18 * f
		top_sign = tf.sign(to_cube_root)
		top = top_sign * (top_sign * to_cube_root) ** (1.0/3.0)
		bottom = 2 * (3 * tf.abs(b)) ** (2.0/3.0) 
		last_term = c/(2 * np.sqrt(3) * b)

		inverse = top/bottom - last_term
		return inverse

	def _inverse_cdf(self, f):
		size = tf.shape(f)[0]
		x_ta = tf.TensorArray(tf.float32, size=size, infer_shape=True)
		
		def inverse(counter, x_ta):
			b = self._b[counter] 
			a = self._a[counter]
			cdf = f[counter]
			finite = tf.greater(tf.abs(b), self._tol)
			x = tf.cond(finite, true_fn=lambda: self._inverse_cdf_finite_b(cdf, a, b),
								false_fn=lambda: self._inverse_cdf_zero_b(cdf, a))
			x_ta = x_ta.write(counter, x)
			return counter+1, x_ta

		cond = lambda c, x: tf.less(c, size)
		_, x_ta = tf.while_loop(cond=cond, 
								body=inverse, 
								loop_vars=[0, x_ta],
								shape_invariants=[tf.TensorShape([]), tf.TensorShape(None)])
		return x_ta.stack()


if __name__ == '__main__':

	a_nparray = np.random.uniform(size=10000, low=-10, high=10).astype(np.float32)
	b_nparray = np.random.uniform(size=10000, low=-10, high=10).astype(np.float32)

	a = tf.constant(a_nparray)
	b = tf.constant(b_nparray)

	quad_dist = Quadratic(a, b)
	samples = quad_dist.sample() 
	cdfs = quad_dist.cdf(samples)

	with tf.Session() as sess:
		tf.global_variables_initializer()
		samples, cdfs = sess.run([samples, cdfs])
		print(samples[samples > 1].shape)
		print(cdfs[cdfs < 0.5].shape)










