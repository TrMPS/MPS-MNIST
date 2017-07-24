from optimizer import *
import MNISTpreprocessing

# Model parameters
d_feature = 2
d_output = 10
batch_size = 1000
permuted = False
shuffled = False
shrink = True
input_size = 784
if shrink:
    input_size = 196

max_size = 20

rate_of_change = 10 ** (-7) 
logging_enabled = False

cutoff = 10 # change this next
n_step = 10

data_source = MNISTpreprocessing.MNISTDatasource(shrink = shrink, permuted = permuted, shuffled = shuffled)

# Initialise the model

#with open('weights', 'rb') as fp:
#    weights = pickle.load(fp)
#    if len(weights) != input_size:
#        weights = None

weights = None

network = MPS(d_feature, d_output, input_size)
network.prepare(data_source)
optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
optimizer.train(data_source, batch_size, n_step, 
                rate_of_change=rate_of_change, 
                _logging_enabled=logging_enabled, 
                initial_weights=weights)
