import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.MNISTpreprocessing as MNISTpreprocessing
from trmps import *
from utils import plot_func

# Model parameters
d_feature = 2
d_output = 11
input_size = 784
lin_reg_learning_rate = 10**(-1)

# Data parameters
permuted = False
shuffled = True
shrink = True
if shrink:
    input_size = 196

special_node_loc = None

# Optimizer parameters
batch_size = 10000
max_size = 20
min_singular_value = 0.001
reg = 0.01
armijo_coeff = 10**(-1)

rate_of_change = 5 * 10 ** (-4)
lr_reg = 0.0

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 6

data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)
# data_source._save_binary()

weights = None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)
# Create network from scratch
network = MPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate, iterations=10000)

# Load network from saved configuration
# network = MPS.from_file()

# Training
optimizer = SingleSiteMPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters, plot_func=plot_func)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
