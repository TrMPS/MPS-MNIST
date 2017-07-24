from optimizer import *
from word2vec import MovieReviewDatasource

# Model parameters
d_feature = 2
d_output = 1
batch_size = 1000
permuted = False
shuffled = False

max_doc_length = 100
embedding_size = 30 
input_size = max_doc_length * embedding_size

max_size = 20

rate_of_change = 10 ** (-7)
logging_enabled = False

cutoff = 10 # change this next
n_step = 10

data_source = MovieReviewDatasource(shuffled = shuffled)
weights = None

network = MPS(d_feature, d_output, input_size)
network.prepare(data_source)
feature, label = data_source.next_training_data_batch(1000)
network.test(feature, label)
# optimizer = MPSOptimizer(network, max_size, None, cutoff=cutoff)
# optimizer.train(data_source, batch_size, n_step, 
#                 rate_of_change=rate_of_change, 
#                 logging_enabled=logging_enabled, 
#                 initial_weights=weights)
