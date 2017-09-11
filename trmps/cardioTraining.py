import cardiopreprocessing
from trmps import *

# Model parameters
d_feature = 2
d_output = 4
input_size = 400
lin_reg_iterations = 1000
special_node_loc = 200
lin_reg_learning_rate = 10**(-4)

# Data parameters
permuted = False
shuffled = False

# Optimizer parameters
reg = 0.1
armijo_iterations = 20
max_size = 40
rate_of_change = 10**(-5)
verbosity = -0
cutoff = 10**(-1)  # change this next
n_step = 300
logging_enabled = False
optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, verbosity=verbosity)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=None,
                                            _logging_enabled=logging_enabled)
optimizer_parameters.armijo_iterations = armijo_iterations


data_source = cardiopreprocessing.cardioDatasource(shuffled=shuffled)
batch_size = int(data_source.num_train_samples)

print(data_source.num_train_samples, data_source.num_test_samples)

weights = None

# DMRG optimizer
# Create network from scratch
network = sqMPS(d_feature, d_output, input_size,
                special_node_loc=special_node_loc)
network.prepare(data_source, lin_reg_iterations)

# Load network from saved configuration
# network = MPS.from_file()

optimizer = MPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)

# Testing
# network.test(data_source.test_data[0], data_source.test_data[1])

