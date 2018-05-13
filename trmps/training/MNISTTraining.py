import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing.MNISTpreprocessing as mp
from trmps import *

# Model parameters
d_feature = 2
d_output = 10
input_size = 784
lin_reg_learning_rate = 10**(-1)

# Data parameters
permuted = False
shuffled = False
shrink = True
if shrink:
    input_size = 196
special_node_loc = None
add_random = False
if add_random:
    d_output = 11

# Optimizer parameters
sweep_range = (1, input_size-2)
batch_size = 10000
max_size = 30
min_singular_value = 1e-8
reg = 1e-2
use_hessian = False

rate_of_change = 1e-6
lr_reg = 0.1
logging_enabled = False
verbosity = 0
updates_per_step = 10
path = "MNISTMPSConfig"

cutoff = 100
n_step = 3

data_source = mp.MNISTDatasource(shrink=shrink,
								 permuted=permuted,
								 shuffled=shuffled,
								 add_random=add_random)
print(data_source.test_data[0].shape)
print(data_source.test_data[1].shape)


optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity, use_hessian=use_hessian,
                                              updates_per_step=updates_per_step,
                                              path=path)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change,
                                            _logging_enabled=logging_enabled)

# Create network from scratch
network = shortMPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate, iterations=10000)

# Load network from saved configuration
# network = shortMPS.from_file()

#ShortMPS training
optimizer = shortMPSOptimizer(network,
						 	  max_size,
							  sweep_range=sweep_range,
							  optional_parameters=optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)

# Training
# optimizer = SingleSiteMPSOptimizer(network, max_size, optimizer_parameters)
# optimizer.train(data_source, batch_size, n_step,
#                 training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])
