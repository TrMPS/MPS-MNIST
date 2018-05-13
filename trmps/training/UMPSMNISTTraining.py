import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.umpsMNISTpreprocessing as pp
from trmps import *

# Model parameters
d_feature = 2
d_output = 102
expected_length = 784
lin_reg_learning_rate = 10**(-1)
initial_bond_dim = 2

# Data parameters
permuted = False
shuffled = False
shrink = True
if shrink:
    expected_length = 196
label_index = 1

# Optimizer parameters
batch_size = 12716
max_size = 30
min_singular_value = 0.001
reg = 0.01

rate_of_change = 5.0 * 10 ** (6)
lr_reg = 0.0

logging_enabled = False
verbosity = -0

cutoff = 100
n_step = 6

data_source = pp.UMPSMNISTDatasource(label_index=label_index, shrink=shrink, permuted=permuted, shuffled=shuffled)
print(data_source.num_train_samples)

weights = None

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)

network = UMPS(expected_length, d_feature, d_output, initial_bond_dim=initial_bond_dim)

# Training
optimizer = UMPSOptimizer(network, max_size, expected_length, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                optional_parameters=training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
