import numpy as np 	
import tensorflow as tf
from tensorflow.python.ops.distributions import distribution


class Quadratic(distribution.Distribution):

	def __init__(self, 
				 a, 
				 b, 
				 tol=1e-4,
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
		parameters = locals() 
		super(Quadratic, self).__init__(
			dtype=self._a.dtype, 
			reparameterization_type=distribution.FULLY_REPARAMETERIZED, 
			validate_args=True, 
			allow_nan_stats=allow_nan_stats, 
			parameters=parameters,
			graph_parents=[self._a, self._b], 
			name=name)

	def _sample_n(self, n, seed=None):
		cdfs = tf.random_uniform([n], minval=0, maxval=1, seed=seed)
		return self._inverse_cdf(cdfs)

	def _prob(self, x):
		u = 2 * x - 1
		middle_term = 2 * np.sqrt(3) * u * self._a * self._b 
		last_term = 3 * tf.square(u) * self._b_sq 
		return self._a_sq + middle_term + last_term

	def _log_prob(self, x):
		return tf.log(self._prob(x))

	def _cdf(self, x):
		first_term = self._a_sq * x 
		second_term = 2 * np.sqrt(3) * self._a * self._b * (x - 1) * x 
		third_term =  self._b_sq * ((2 * x - 1) ** 3 + 1)/2 
		return first_term + second_term + third_term

	def _log_cdf(self, x):
		return tf.log(self._cdf(x))

	def _mean(self, x): 
		return self._a * self._b/np.sqrt(3) 

	def _inverse_cdf_zero_b(self, f):
		return f/tf.square(self._a - np.sqrt(3) * self._b)

	def _inverse_cdf_finite_b(self, f):
		c = self._a - np.sqrt(3) * self._b
		to_cube_root = np.sqrt(3) * c ** 3 / self._b + 18 * f
		sign = tf.sign(to_cube_root)
		top = sign * (sign * to_cube_root) ** (1.0/3.0)
		bottom = 2 * (3 * self._b) ** (2.0/3.0) 
		last_term = c/(2 * np.sqrt(3) * self._b)
		return top/bottom - last_term

	def _inverse_cdf(self, f):
		x = tf.cond(tf.less(self._b, self._tol), 
					true_fn=lambda: self._inverse_cdf_zero_b(f), 
					false_fn=lambda: self._inverse_cdf_finite_b(f))
		return x 


if __name__ == '__main__':

	a = tf.constant(np.cos(1).astype(np.float32))
	b = tf.constant(np.sin(1).astype(np.float32))

	n_samples = 10000
	quad_dist = Quadratic(a, b)
	samples = quad_dist.sample(n_samples) 
	empirical_cdf = tf.shape(tf.boolean_mask(samples, tf.less(samples, 0.5)))[0]/n_samples
	analytical_cdf = quad_dist.cdf(0.5)  


	with tf.Session() as sess:
		tf.global_variables_initializer()
		print(sess.run([empirical_cdf, analytical_cdf]))










