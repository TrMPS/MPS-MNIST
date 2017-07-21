from optimizer import *
import activitypreprocessing as ap

# Model parameters
d_feature = 4
d_output = 7
batch_size = 1000
permuted = False
shuffled = False
input_size = 200

max_size = 20

rate_of_change = 10 ** (-7)
logging_enabled = False

cutoff = 10 # change this next
n_step = 10

data_source = ap.activityDatasource(shuffled = shuffled)

print(data_source.num_train_samples, data_source.num_test_samples)

weights = None

network = MPS(d_feature, d_output, input_size)
network.prepare(data_source)
optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
optimizer.train(data_source, batch_size, n_step, 
                rate_of_change=rate_of_change, 
                logging_enabled=logging_enabled, 
                initial_weights=weights)
