import tensorflow as tf
import time
from mps_sgd import SimpleMPS
import utils
from tensorflow.python.client import timeline
from MNISTpreprocessing import MNISTDatasource

class SGDOptimizer(object):

    def __init__(self, network):
        self.MPS = network

    def train(self, data_source, batch_size, n_steps, rate_of_change=0.001):
        features = tf.placeholder(tf.float32, shape=[self.MPS.input_size, None, self.MPS.d_feature])
        labels = tf.placeholder(tf.float32, shape=[None, self.MPS.d_output])

        f = self.MPS.predict(features)
        cost = self.MPS.cost(f, labels)
        accuracy = self.MPS.accuracy(f, labels)
        conf_mat = self.MPS.confusion_matrix(f, labels)
        trainer = tf.train.AdamOptimizer(rate_of_change).minimize(cost)

        to_eval = [accuracy, cost, conf_mat]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(n_steps):
                batch_feature, batch_label = data_source.next_training_data_batch(batch_size)
                feed_dict = {features: batch_feature, labels: batch_label}
                _ = sess.run(trainer, feed_dict=feed_dict)
                if i % 10 == 0:
                    train_acc, train_cost, train_conf_mat = sess.run(to_eval, feed_dict=feed_dict)
                    print('step {}, accuracy {}'.format(i, train_acc))
                    print('step {}, training cost {}'.format(i, train_cost))
                    print(train_conf_mat)

            test_feature, test_label = data_source.test_data
            feed_dict = {features: test_feature, labels: test_label}
            to_eval = [self.MPS.nodes, accuracy, cost, conf_mat]
            weights, test_acc, test_cost, test_conf_mat = sess.run(to_eval, feed_dict=feed_dict)
            print('testing accuracy {}'.format(test_acc))
            print('testing cost {}'.format(test_cost))
            print(test_conf_mat)

            with open('weights_sgd', 'wb') as fp:
                pickle.dump(weights, fp)

if __name__ == '__main__':
    # Model parameters
    d_feature = 2
    d_output = 10
    batch_size = 100
    permuted = False
    shuffled = True
    shrink = True
    input_size = 784
    if shrink:
        input_size = 196

    rate_of_change = 0.001
    feature_reg=1.1
    reg=0.001
    n_step = 1000

    data_source = MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

    weights = None

    network = SimpleMPS(d_feature, d_output, input_size, 
                        feature_reg=feature_reg, 
                        reg=reg)
    network.prepare(data_source)
    optimizer = SGDOptimizer(network)
    optimizer.train(data_source, batch_size, n_step, 
                    rate_of_change=rate_of_change)


        
