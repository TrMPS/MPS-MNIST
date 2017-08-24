from short_optimizer import shortMPSOptimizer
from singlesiteOptimizer import *
import MNISTpreprocessing
from short_mps import shortMPS
from squaredDistanceMPS import *

# Model parameters
d_feature = 2
d_output = 10
input_size = 784

# Data parameters
permuted = False
shuffled = True
shrink = True
if shrink:
    input_size = 196
special_node_loc = 91

# Optimizer parameters
batch_size = 2000
max_size = 30
min_singular_value = 1e-8
reg = 1e-2
use_hessian = False

rate_of_change = 1e-5
lr_reg = 0.1
logging_enabled = False
verbosity = 100

cutoff = 100
n_step = 12

data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

# Initialise the model

with open('weights', 'rb') as fp:
    weights = pickle.load(fp)
    if len(weights) != input_size:
        weights = None

# network = shortMPS(d_feature, d_output, input_size, special_node_loc)
# network.prepare(data_source=data_source)
# weights=None
# optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
#                                               verbosity=verbosity, use_hessian=use_hessian)
# training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
#                                             _logging_enabled=logging_enabled)
# optimizer = shortMPSOptimizer(network, max_size, optimizer_parameters)
# optimizer.train(data_source, batch_size, n_step,
#                 training_parameters)

network = shortMPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=None)
feed_dict = network.create_feed_dict(weights)
test_features, test_labels = data_source.test_data
features = tf.placeholder(tf.float32, shape=[input_size, None, d_feature])
labels = tf.placeholder(tf.float32, shape=[None, d_output])
f = network.predict(features)
confusion_matrix = network.confusion_matrix(f, labels)
accuracy = network.accuracy(f, labels)
feed_dict[features] = test_features
feed_dict[labels] = test_labels
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conf, acc, test_f = sess.run([confusion_matrix, accuracy, f], feed_dict = feed_dict)
print("\n\n\n\n Accuracy is:" + str(acc))
print("\n\n\n\n" + str(conf))
print("\n\n\n\n Sample prediction: " + str(test_f[0]))

