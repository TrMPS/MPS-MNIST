import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.randomsequencepreprocessing as rs
from trmps import *

# Data parameters
sequence_length = 30
num_bracket_types = 5
num_noise_types = num_bracket_types
max_unmatched = 10
num_samples = 1500
normalised = False
encoding = rs.inputEncoding.linear

# Model parameters
d_feature = 2
d_output = 2
input_size = sequence_length
lin_reg_learning_rate = 10**(-3)
lin_reg_iterations = 20000

special_node_loc = None

# Optimizer parameters
batch_size = num_samples
max_size = 5
min_singular_value = 0.001
reg = 0.01
armijo_coeff = 10**(-1)
updates_per_step = 20
verbose_save = True

rate_of_change = 10**(-2)
lr_reg = 0.0

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 200

data_source = rs.RandomSequenceDatasource(sequence_length, encoding, num_samples)
# data_source._save_binary()

weights = None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity, armijo_coeff=armijo_coeff, updates_per_step=updates_per_step)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled, verbose_save=True)
# Create network from scratch
network = MPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate, iterations=lin_reg_iterations)
# network.prepare(data_source=None)

# Load network from saved configuration
# network = MPS.from_file()

# Training
optimizer = SingleSiteMPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
