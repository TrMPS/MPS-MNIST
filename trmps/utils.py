import tensorflow as tf
import urllib
import tarfile

def check_nan(tensor, name, replace_nan=True):
    s = tf.reduce_sum(tensor)
    is_nan = tf.is_nan(s)
    tensor = tf.cond(is_nan, 
                     true_fn=lambda: tf.Print(tensor, [tensor], 
                                            message='{} is not finite'.format(name)), 
                     false_fn=lambda: tensor)
    if replace_nan:
        tensor = tf.where(tf.is_nan(tensor), 
                          tf.zeros_like(tensor), 
                          tensor)
    return tensor 

def getunzipped(url, name):
    try:
        name, hdrs = urllib.request.urlretrieve(url, name)
    except IOError as e:
        print('Cannot retrieve {}: {}'.format(url, e))
        return 
    z = tarfile.open(name, "r:gz")
    z.extractall()
    z.close()
    print('Data downloaded and unzipped')








