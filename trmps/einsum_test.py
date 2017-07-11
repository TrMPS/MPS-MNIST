import tensorflow as tf 
import numpy as np 

batch_size = 1
d_feature = 3
d_matrix = 2


C1 = tf.Variable(np.ones([batch_size, d_matrix]))
C2 = tf.Variable(np.ones([batch_size, d_matrix]))

input1 = tf.Variable(np.ones([batch_size, d_feature]))
input2 = tf.Variable(np.ones([batch_size, d_feature]))

# C = tf.einsum('ti,tk,tm,tn->tmnik', C1, C2, self._feature[counter], self._feature[counter + 1])

C1 = tf.reshape(C1, [batch_size, 1, 1, d_matrix, 1])
C2 = tf.reshape(C2, [batch_size, 1, 1, 1, d_matrix])
input1 = tf.reshape(input1, [batch_size, d_feature, 1, 1, 1])
input2 = tf.reshape(input2, [batch_size, 1, d_feature, 1, 1])

result = C1 * C2 * input1 * input2

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(result))