import tensorflow as tf

def _test_fin(tensor):
    s = tf.reduce_sum(tensor)
    return tf.is_finite(s)

