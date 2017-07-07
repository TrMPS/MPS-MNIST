import tensorflow as tf 
import numpy as np

class TensorArrayTest(object):

	def __init__(self):
		self.ta = tf.TensorArray(dtype=tf.float32, 
									tensor_array_name='tensorarray', 
									size=3, infer_shape=False)

		A = np.arange(4).reshape(2, 2).astype(np.float32)
		B = np.arange(8).reshape(2, 4).astype(np.float32)
		C = np.arange(6).reshape(2, 3).astype(np.float32)
		self.ta = self.ta.write(0, A)
		self.ta = self.ta.write(1, B)
		self.ta = self.ta.write(2, C)


	def update(self):
		cond = lambda counter, b: counter < 3
		tb = tf.TensorArray(dtype=tf.float32, size=3, infer_shape=False)
		_, tb = tf.while_loop(cond=cond, body=self._one_loop, 
									loop_vars=[0, tb],
									parallel_iterations=1, 
									shape_invariants=[tf.TensorShape([]),
													  tf.TensorShape(None)])
		return self.read()
	
	def _one_loop(self, counter, tb):


		tb = tb.write(counter, self.ta.read(counter) + 1)
		counter += 1 

		return [counter, tb]
	
	def read(self):
		r = [self.ta.read(i) for i in range(3)]
		return r 


if __name__ == '__main__':
	test = TensorArrayTest()
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		writer = tf.summary.FileWriter("output", sess.graph)
		writer.close()

		r = test.update()
		output = sess.run(r)
		print(output)

		writer = tf.summary.FileWriter("output", sess.graph)
		writer.close()










		