from TensorGANOptimizer import TensorGANOptimizer
from TensorGAN import GeneratorMPS
import MNISTpreprocessing
from mps import MPS

# Common parameters
batch_size = 10000
permuted = False
shuffled = True
shrink = True
mnist_input_size = 784
if shrink:
    mnist_input_size = 196
min_singular_value = 0.01
reg = 0.01

# Generator parameters
d_feature = 2
seed_length = 100
d_output = mnist_input_size
special_node_loc = int(seed_length / 2)
generator_max_size = 10
generator = GeneratorMPS(d_feature, d_output, seed_length, special_node_loc)
generator.prepare(data_source=None)

# Discriminator parameters
d_feature = 2
d_output = 2
input_size = mnist_input_size
special_node_loc = int(mnist_input_size / 2)
discriminator_max_size = 15
discriminator = MPS(d_feature, d_output, input_size, special_node_loc=None)
discriminator.prepare(data_source=None)

generator_rate = 10 ** (-6)
discriminator_rate = 10** (-6)
logging_enabled = False
verbose = 0

cutoff = 100
n_step = 1000

data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

optimizer = TensorGANOptimizer(generator, discriminator, generator_max_size, discriminator_max_size)
optimizer.train(data_source, n_step, generator_rate, discriminator_rate,
              initial_generator_weights = None, initial_discriminator_weights = None)
