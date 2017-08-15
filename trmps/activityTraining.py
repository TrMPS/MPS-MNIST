from optimizer import *
import activitypreprocessing as ap

# Model parameters
d_feature = 4
d_output = 7
batch_size = 2000
permuted = False
shuffled = False
input_size = 100
lin_reg_iterations = 1000
special_node_loc = 50

max_size = 15

rate_of_change = 5*10**(-5)
lr_reg = 0.0
reg = 0.001
logging_enabled = False

cutoff = 10  # change this next
n_step = 300
verbose = 0

data_source = ap.activityDatasource(shuffled=shuffled)
# batch_size = data_source.num_train_samples

print(data_source.num_train_samples, data_source.num_test_samples)

# Training

weights = None

network = MPS(d_feature, d_output, input_size, special_node_loc=special_node_loc)
network.prepare(data_source=None, iterations=lin_reg_iterations)
feature, label = data_source.next_training_data_batch(1000)
# network.test(feature, label)
optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff,
                         reg=reg, lr_reg=lr_reg,
                         verbose=verbose)
optimizer.train(data_source, batch_size, n_step,
                rate_of_change=rate_of_change,
                _logging_enabled=logging_enabled,
                initial_weights=weights)

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
