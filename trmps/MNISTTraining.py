from optimizer import *
import MNISTpreprocessing
from mps import * 

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
batch_size = 10000
max_size = 20 
min_singular_value = 1e-8
reg = 1e-2
use_hessian = False

rate_of_change = 1e-5
lr_reg = 0.0 
logging_enabled = False
verbosity = 50000

cutoff = 100
n_step = 7

data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

# Initialise the model

# with open('weights_sgd', 'rb') as fp:
#     weights = pickle.load(fp)
#     if len(weights) != input_size:
#         weights = None

weights=None
optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                              verbosity=verbosity, use_hessian=use_hessian)
training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                            _logging_enabled=logging_enabled)
network = MPS(d_feature, d_output, input_size, special_node_loc)
network.prepare(data_source=None)
optimizer = MPSOptimizer(network, max_size, optimizer_parameters)
optimizer.train(data_source, batch_size, n_step,
                training_parameters)