from FashionMNISTpreprocessing import FashionMNISTDatasource
from trmps import *

# Model parameters
d_feature = 2
d_output = 10
input_size = 784
lin_reg_learning_rate = 10**(-4)
lin_reg_iterations = 10000

# Data parameters
permuted = False
shuffled = False
shrink = False
if shrink:
    input_size = 196

special_node_loc = 14

# Optimizer parameters
batch_size = 20000
max_size = 25
min_singular_value = 0.00
reg = 0.01
armijo_coeff = 10**(-4)
armijo_iterations = 25

rate_of_change = 5 * 10 ** (-4)
lr_reg = 0.0

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 6

data_source = FashionMNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)
network = sqMPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=None, learning_rate=lin_reg_learning_rate,
                iterations=lin_reg_iterations)
# Load weights
with open('weights', 'rb') as fp:
    weights = pickle.load(fp)
    if len(weights) != input_size:
        weights = None

# weights=None
# optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
#                                               verbosity=verbosity, armijo_iterations=armijo_iterations)
# training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
#                                             _logging_enabled=logging_enabled)
# optimizer = SingleSiteMPSOptimizer(network, max_size, optimizer_parameters)
# optimizer.train(data_source, batch_size, n_step,
#                 training_parameters)

# Testing

feed_dict = network.create_feed_dict(weights)
test_features, test_labels = data_source.test_data
features = tf.placeholder(tf.float32, shape=[input_size, None, d_feature])
labels = tf.placeholder(tf.float32, shape=[None, d_output])
f = network.predict(features)
confusion_matrix = network.confusion_matrix(f, labels)
accuracy = network.accuracy(f, labels)
cost = network.cost(f, labels)
feed_dict[features] = test_features
feed_dict[labels] = test_labels
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conf, acc, test_f, test_cost = sess.run([confusion_matrix, accuracy, f, cost], feed_dict = feed_dict)
print("\n\n\n\n Accuracy is:" + str(acc) + " and cost is:" + str(test_cost))
print("\n\n\n\n" + str(conf))
print("\n\n\n\n Sample prediction: " + str(test_f[1]))
print("Correct prediction for this case is: " + str(test_labels[1]))
