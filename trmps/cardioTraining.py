from optimizer import *
import cardiopreprocessing
from sgd_optimizer import SGDOptimizer
from mps_sgd import SimpleMPS
from squaredDistanceMPS import *

# Model parameters
d_feature = 2
d_output = 4
input_size = 400
lin_reg_iterations = 1000
special_node_loc = 200
lin_reg_learning_rate = 10**(-4)

# Data parameters
permuted = False
shuffled = False

# Optimizer parameters
reg = 0.1
max_size = 23
rate_of_change = 10**(-5)
verbosity = -0
cutoff = 10**(-1)  # change this next
n_step = 300
logging_enabled = False
optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=None,
                                            _logging_enabled=logging_enabled)


data_source = cardiopreprocessing.cardioDatasource(shuffled=shuffled)
batch_size = int(data_source.num_train_samples)

print(data_source.num_train_samples, data_source.num_test_samples)

weights = None

# DMRG optimizer

with open('weights_sgd', 'rb') as fp:
    weights = pickle.load(fp)
    if len(weights) != input_size:
        weights = None
training_parameters.initial_weights = weights

network = sqMPS(d_feature, d_output, input_size,
                special_node_loc=special_node_loc)
network.prepare(None, lin_reg_iterations)
optimizer = sqMPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)

# SGD optimizer

# batch_size = int(data_source.num_train_samples / 10)
# network = SimpleMPS(d_feature, d_output, input_size,
#                     special_node_loc=special_node_loc, reg=reg)
# network.prepare(data_source, lin_reg_iterations)
# optimizer = SGDOptimizer(network)
# optimizer.train(data_source, batch_size, n_step,
#                 rate_of_change=rate_of_change)

# Testing

# with open('weights', 'rb') as fp:
#    weights = pickle.load(fp)
#    if len(weights) != input_size:
#        weights = None
#
# network.prepare(data_source)
# feed_dict = network.create_feed_dict(weights)
# test_features, test_labels = data_source.test_data
# features = tf.placeholder(tf.float32, shape=[input_size, None, d_feature])
# labels = tf.placeholder(tf.float32, shape=[None, d_output])
# f = network.predict(features)
# confusion_matrix = network.confusion_matrix(f, labels)
# accuracy = network.accuracy(f, labels)
# feed_dict[features] = test_features
# feed_dict[labels] = test_labels
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     conf, acc = sess.run([confusion_matrix, accuracy], feed_dict = feed_dict)
# print("\n\n\n\n Accuracy is:" + str(acc))
# print("\n\n\n\n" + str(conf))
