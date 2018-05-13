import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.bracketmatchingpreprocessing as bmp
from trmps import *

# Data parameters
sequence_length = 50
num_bracket_types = 10
num_noise_types = num_bracket_types
max_unmatched = 10000
num_test_data = 60000
num_train_data = 20000
normalised = False

# Model parameters
d_feature = (num_bracket_types * 2) + num_noise_types + 1
d_output = num_bracket_types
input_size = sequence_length
lin_reg_learning_rate = 10**(-3)
lin_reg_iterations = 20000
round = True
updates_per_step = 10

special_node_loc = 1

# Optimizer parameters
batch_size = 1000
max_size = 60
min_singular_value = 0.001
reg = 0.01
armijo_coeff = 0.0

rate_of_change = 1
lr_reg = 0.0
path = "BracketMatchingMPSConfig"

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 5

data_source = bmp.BracketMatchingDatasource(sequence_length, num_bracket_types, num_noise_types,
                                            max_unmatched, num_test_data, num_train_data, normalised)
# data_source._save_binary()

weights = None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity, armijo_coeff=armijo_coeff,
                                              updates_per_step=updates_per_step,
                                              path=path)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)
# Create network from scratch
network = sqMPS(d_feature, d_output, input_size, special_node_loc, round=round)
network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate, iterations=lin_reg_iterations)
# network.prepare(data_source=None)

# Load network from saved configuration
# network = MPS.from_file()
# network.round = round

# Training
optimizer = SingleSiteMPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
