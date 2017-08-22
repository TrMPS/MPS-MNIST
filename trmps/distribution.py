import numpy as np 	
import tensorflow as tf
from tensorflow.python.ops.distributions import distribution


class Quadratic(object):

	def __init__(self, 
				 a, 
				 b, 
				 tol=1e-6,
				 allow_nan_stats=False, 
				 name="quadratic"):
		norm = a * a + b * b 
		with tf.control_dependencies([tf.assert_greater(norm, 1 - tol),
									  tf.assert_less(norm, 1 + tol)]):
			self._a = a 
			self._b = b 
			self._a_sq = a * a 
			self._b_sq = b * b 
			self._tol = tol
			self._n_params = tf.shape(a)[0] 

	def sample(self, seed=None):
		cdfs = tf.random_uniform([self._n_params], minval=0, maxval=1, seed=seed)
		return self._inverse_cdf(cdfs)

	def prob(self, x):
		u = 2 * x - 1
		middle_term = 2 * np.sqrt(3) * u * self._a * self._b 
		last_term = 3 * tf.square(u) * self._b_sq 
		return self._a_sq + middle_term + last_term

	def cdf(self, x):
		first_term = self._a_sq * x 
		second_term = 2 * np.sqrt(3) * self._a * self._b * (x - 1) * x 
		third_term =  self._b_sq * ((2 * x - 1) ** 3 + 1)/2 
		return first_term + second_term + third_term

	def _mean(self, x): 
		return self._a * self._b/np.sqrt(3) 

	def _inverse_cdf_zero_b(self, f, a, b):
		return f/tf.square(a - np.sqrt(3) * b)

	def _inverse_cdf_finite_b(self, f, a, b):
		c = a - np.sqrt(3) * b
		to_cube_root = np.sqrt(3) * c ** 3 / b + 18 * f
		sign = tf.sign(to_cube_root)
		top = sign * (sign * to_cube_root) ** (1.0/3.0)
		bottom = 2 * (3 * b) ** (2.0/3.0) 
		last_term = c/(2 * np.sqrt(3) * b)
		return top/bottom - last_term

	def _inverse_cdf(self, f):
		size = tf.shape(f)[0]
		x_ta = tf.TensorArray(tf.float32, size=size, infer_shape=True)
		
		def inverse(counter, x_ta):
			b = self._b[counter] 
			a = self._a[counter]
			cdf = f[counter]
			finite = tf.greater(b, self._tol)
			x = tf.cond(finite, true_fn=lambda: self._inverse_cdf_finite_b(cdf, a, b),
								false_fn=lambda: self._inverse_cdf_zero_b(cdf, a, b))
			x_ta = x_ta.write(counter, x)
			return counter+1, x_ta
		cond = lambda c, x: tf.less(c, size)

		_, x_ta = tf.while_loop(cond=cond, 
								body=inverse, 
								loop_vars=[0, x_ta],
								shape_invariants=[tf.TensorShape([]), tf.TensorShape(None)])
		return x_ta.stack()


if __name__ == '__main__':

	theta = np.random.uniform(size=100).astype(np.float32)
	a = tf.constant(np.cos(theta))
	b = tf.constant(np.sin(theta))

	quad_dist = Quadratic(a, b)
	samples = quad_dist.sample() 
	cdfs = quad_dist.cdf(samples)
	count = tf.shape(tf.boolean_mask(cdfs, tf.less(cdfs, 0.5)))


	with tf.Session() as sess:
		tf.global_variables_initializer()
		print(sess.run([count]))










