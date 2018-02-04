import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.rmpsMNISTpreprocessing as pp
from trmps import *

# Model parameters
d_feature = 2
d_output = 10
expected_length = 784
lin_reg_learning_rate = 10**(-1)

# Data parameters
permuted = False
shuffled = False
shrink = True
if shrink:
    expected_length = 196

# Optimizer parameters
batch_size = 10000
max_size = 30
min_singular_value = 0.001
reg = 0.01

rate_of_change = 5 * 10 ** (-4)
lr_reg = 0.0

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 6

data_source = pp.RMPSMNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

weights = None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)
# Create network from scratch
# network = MPS(d_feature, d_output, input_size, special_node_loc)
# network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate, iterations=10000)

# Load network from saved configuration
network = RMPS(expected_length, d_feature, d_output)

# Training
optimizer = RMPSOptimizer(network, max_size, expected_length, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                optional_parameters=training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
