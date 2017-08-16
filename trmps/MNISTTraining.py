from optimizer import *
import MNISTpreprocessing
from MPS import *

# Model parameters
d_feature = 2
d_output = 10

input_size = 784
lin_reg_learning_rate = 10**(-4)

# Data parameters
permuted = False
shuffled = True
shrink = True
if shrink:
    input_size = 196

special_node_loc = 98


# Optimizer parameters
batch_size = 2000
max_size = 30
min_singular_value = 0.001

reg = 0.01
armijo_coeff = 10**(-1)

rate_of_change = 10 ** (-4)

lr_reg = 0.0
armijo_coeff = 10 ** (-4)

logging_enabled = False

verbosity = -0


cutoff = 100
n_step = 6

data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

# Initialise the model

# with open('weights_sgd', 'rb') as fp:
#     weights = pickle.load(fp)
#     if len(weights) != input_size:
#         weights = None

weights=None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)

network = MPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate)
optimizer = MPSOptimizer(network, max_size, optimizer_parameters)

optimizer.train(data_source, batch_size, n_step,
                training_parameters)

