from optimizer import *
from word2vec import MovieReviewDatasource

# Model parameters
d_feature = 2
d_output = 8
batch_size = 5000
shuffled = True

max_length_for_skip_gram = 200
max_length_for_mps = 50 
embedding_size = 80
expected_shape = (max_length_for_mps*embedding_size, 2)


max_size = 16


lin_reg_iterations = 250 
lin_reg_lr = 0.02
rate_of_change = 5 * 10 ** (-5) 
logging_enabled = False

cutoff = 10000 
n_step = 5

datasource = MovieReviewDatasource(expected_shape=expected_shape, 
                                   embedding_size=embedding_size, 
                                   max_length_for_skip_gram=max_length_for_skip_gram,
                                   max_length_for_mps=max_length_for_mps)

weights = None
network = MPS(d_feature, d_output, input_size)
network.prepare(data_source, lin_reg_iterations, lin_reg_lr)
optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)

data_source.shuffle()
optimizer.train(data_source, batch_size, n_step, 
                rate_of_change=rate_of_change, 
                logging_enabled=logging_enabled, 
                initial_weights=weights)

