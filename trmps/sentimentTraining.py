from optimizer import *
from word2vec import MovieReviewDatasource

# Model parameters
d_feature = 2
d_output = 8
batch_size = 5000
shuffled = True


max_doc_length = 100
embedding_size = 30 
input_size = max_doc_length * embedding_size

expected_shape = (input_size, d_feature)

max_size = 18

rate_of_change = 5 * 10 ** (-5) 
logging_enabled = False

cutoff = 10000 
n_step = 5

data_source = MovieReviewDatasource(shuffled=shuffled, 
									embedding_size=embedding_size,
									expected_shape=expected_shape)
weights = None
network = MPS(d_feature, d_output, input_size)
network.prepare(data_source)
optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)

data_source.reshuffle()
optimizer.train(data_source, batch_size, n_step, 
                rate_of_change=rate_of_change, 
                _logging_enabled=logging_enabled, 
                initial_weights=weights)

