import tensorflow as tf
import numpy as np
from mps import MPS
import utils
from generatorOptimizer import generatorOptimizer
from optimizer import MPSOptimizer
from parameterObjects import *

class TensorGANOptimizer(object):
    def __init__(self, generator, discriminator, generator_max_size, discriminator_max_size,
                 generator_parameters=MPSOptimizerParameters(),
                 discriminator_parameters=MPSOptimizerParameters()):
        self.generator = generator
        self.discriminator = discriminator
        self._true_features = tf.placeholder(tf.float32, shape=[self.discriminator.input_size, None, self.discriminator.d_feature])
        self.batch_size = tf.shape(self._true_features)[1]
        ones = tf.ones([self.batch_size, 1])
        zeros = tf.zeros([self.batch_size, 1])
        self._true_labels = tf.concat([ones, zeros], 1)
        print("true labels shape")
        print(self._true_labels.shape)
        self._true_labels = tf.Print(self._true_labels, [tf.shape(self._true_labels)], message="trueLabelsShape")
        self._desired_labels = self._true_labels
        self._false_labels = tf.concat([zeros, ones], 1)
        self._label = tf.concat([self._true_labels, self._false_labels], 0)
        self.generator_optimizer = generatorOptimizer(generator, generator_max_size, self, generator_parameters)
        self.discriminator_optimizer = MPSOptimizer(discriminator, discriminator_max_size, discriminator_parameters)
        self.discriminator_rate = tf.placeholder(tf.float32, shape=[])
        self.generator_rate = tf.placeholder(tf.float32, shape = [])
        self.generator_optimizer.rate_of_change = self.generator_rate
        self.discriminator_optimizer.rate_of_change = self.discriminator_rate
        self.tree_made = False
        # self._true_labels = tf.placeholder(tf.float32, shape=[None, self.discriminator.d_output])
        self.generator_optimizer._label = self._label
        self.discriminator_optimizer._label = self._desired_labels

    def train_step(self):
        self.seed = tf.random_uniform([self.generator.input_size, self.batch_size, self.generator.d_feature])
        xs = self.generator.create(self.seed)
        _phi_part = tf.ones([self.batch_size, self.generator.d_output])
        false_features = tf.stack([_phi_part, xs])
        false_features = tf.transpose(false_features, [2, 1, 0])
        self.discriminator_features = tf.concat([self._true_features, false_features], axis=1)
        self.discriminator_optimizer._feature = self.discriminator_features
        self.generator_optimizer._feature = self.seed
        self.discriminator_optimizer.train_step()
        self.generator_optimizer.train_step()
        self.tree_made = True

    def train(self, data_source, n_step, generator_rate, discriminator_rate, batch_size=1000,
              initial_generator_weights=None, initial_discriminator_weights=None):
        if not self.tree_made:
            self.train_step()
            test_acc, test_sample = self._test_step(self.discriminator_features, self._label, self.batch_size)
        self.generator_weights = initial_generator_weights
        self.discriminator_weights = initial_discriminator_weights
        trained_generator_weights = utils.list_from(self.generator.nodes, self.generator.input_size)
        trained_discriminator_weights = utils.list_from(self.discriminator.nodes, self.discriminator.input_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_step):
                (batch_feature, batch_label) = data_source.next_training_data_batch(batch_size)
                self.feed_dict = self.discriminator.create_feed_dict(self.discriminator_weights)
                self.feed_dict.update(self.generator.create_feed_dict(self.generator_weights))
                self.feed_dict[self._true_features] = batch_feature
                self.feed_dict[self._true_labels] = batch_label
                self.feed_dict[self.discriminator_rate] = discriminator_rate
                self.feed_dict[self.generator_rate] = generator_rate
                to_eval = [trained_discriminator_weights, trained_generator_weights, test_acc, test_sample]
                self.discriminator_weights, self.generator_weights, accuracy, sample = sess.run(to_eval,
                                                                              feed_dict = self.feed_dict)
                print('Discriminator accuracy:', accuracy)
                with open('generatorWeights', 'wb') as fp:
                    pickle.dump(self.generator_weights, fp)
                with open('discriminatorWeights', 'wb') as fp:
                    pickle.dump(self.discriminator_weights, fp)

    def _test_step(self, feature, label, batch_size):
        seed = tf.random_uniform([self.generator.input_size, batch_size, self.generator.d_feature])
        xs = self.generator.create(self.seed)
        _phi_part = tf.ones([self.batch_size, self.generator.d_output])
        false_features = tf.stack([_phi_part, xs])
        false_features = tf.transpose(false_features, [2, 1, 0])
        predictions_true = self.discriminator.predict(feature)
        predictions_false = self.discriminator.predict(false_features)
        accuracy_true = self.discriminator.accuracy(predictions_true, self._true_labels)
        accuracy_false = self.discriminator.accuracy(predictions_false, self._false_labels)
        accuracy = (accuracy_true + accuracy_false)*0.5
        sample = tf.transpose(false_features, perm = [1, 0, 2])[0]
        return accuracy, sample
