import activitypreprocessing as ap
from trmps import *

# Model parameters
d_feature = 4
d_output = 7
input_size = 100
lin_reg_iterations = 10000
special_node_loc = 50
lin_reg_learning_rate = 10**(-5)

# Data parameters
permuted = False
shuffled = False

# Optimizer parameters
max_size = 15
batch_size = 2000
rate_of_change = 10**(-7)
lr_reg = 0.0
reg = 0.001
logging_enabled = False
armijo_coeff = 10**(-1)

cutoff = 10  # change this next
n_step = 300
verbosity = -0

data_source = ap.activityDatasource(shuffled=shuffled)
# batch_size = data_source.num_train_samples

print(data_source.num_train_samples, data_source.num_test_samples)

# Create network from scratch
# network = MPS(d_feature, d_output, input_size, special_node_loc=special_node_loc)
# network.prepare(data_source=data_source, iterations=lin_reg_iterations, learning_rate=lin_reg_learning_rate)

# Network from previous weights
network = MPS.from_file()

network.test(data_source.test_data[0], data_source.test_data[1])

# Training

optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg, verbosity=verbosity,
                                              armijo_coeff=armijo_coeff)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change,
                                            _logging_enabled=logging_enabled)

feature, label = data_source.next_training_data_batch(1000)
optimizer = MPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)
