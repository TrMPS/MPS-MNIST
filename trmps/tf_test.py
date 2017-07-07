import tensorflow as tf
import numpy as np 

a = tf.Variable(0)
b = a 

a += 1 

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(b.eval())