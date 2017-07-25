from optimizer import *
import cardiopreprocessing
from sgd_optimizer import SGDOptimizer
from mps_sgd import SimpleMPS

# Model parameters
d_feature = 2
d_output = 4
batch_size = 5000
permuted = False
shuffled = False
input_size = 900
lin_reg_iterations = 500

max_size = 25

rate_of_change = 10**(-6)
logging_enabled = False

cutoff = 10 # change this next
n_step = 1000

data_source = cardiopreprocessing.cardioDatasource(shuffled = shuffled)
batch_size = int(data_source.num_train_samples/10)

print(data_source.num_train_samples, data_source.num_test_samples)

# Training

weights = None

# DMRG optimizer
# network = MPS(d_feature, d_output, input_size)
# network.prepare(data_source, lin_reg_iterations)
# optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
# optimizer.train(data_source, batch_size, n_step, 
#                 rate_of_change=rate_of_change, 
#                 _logging_enabled=logging_enabled, 
#                 initial_weights=weights)

# SGD optimizer
network = SimpleMPS(d_feature, d_output, input_size)
network.prepare(data_source, lin_reg_iterations)
optimizer = SGDOptimizer(network)
optimizer.train(data_source, batch_size, n_step,
                rate_of_change = rate_of_change)
                
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