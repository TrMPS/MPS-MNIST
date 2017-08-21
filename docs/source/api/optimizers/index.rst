Optimizers
====================================

Introduction to optimizing Matrix Product States for Machine Learning
--------------------------

Following the approach in the paper Supervised learning with Quantum-Inspired Tensor Networks by E.Miles Stoudenmire and David J.Schwab, the MPSOptimizer class and its subclasses optimize the Matrix Product States using two-site DMRG. (Unless explicitly stated otherwise in the subclass). In addition to replicating the behaviour from the paper, the optimizers can also optionally calculate the Hessian, which should theoretically allow the results to converge faster, using a method similar to Newton-Raphson, just extended to higher dimensions.

A simple example of usage is as below:

.. code-block:: python

    from optimizer import *
    import MNISTpreprocessing

    # Model parameters
    d_feature = 2
    d_output = 10
    input_size = 784
    lin_reg_learning_rate = 10**(-4)

    # Data parameters
    permuted = False
    shuffled = True
    shrink = True
    if shrink:
        input_size = 196

    special_node_loc = 98

    # Optimizer parameters
    batch_size = 2000
    max_size = 30
    min_singular_value = 0.001
    reg = 0.01
    armijo_coeff = 10**(-1)

    rate_of_change = 5 * 10 ** (-4)
    lr_reg = 0.0

    logging_enabled = False
    verbosity = -0

    cutoff = 100
    n_step = 6

    data_source = MNISTpreprocessing.MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

    weights=None
    optimizer_parameters = MPSOptimizerParameters(cutoff=cutoff, reg=reg, lr_reg=lr_reg,
                                                  verbosity=verbosity)
    training_parameters = MPSTrainingParameters(rate_of_change=rate_of_change, initial_weights=weights,
                                                _logging_enabled=logging_enabled)

    network = MPS(d_feature, d_output, input_size, special_node_loc)
    network.prepare(data_source=data_source, learning_rate=lin_reg_learning_rate)
    optimizer = MPSOptimizer(network, max_size, optimizer_parameters)
    optimizer.train(data_source, batch_size, n_step,
                    training_parameters)

The example above involves the training of a model for the MNIST dataset. The bulk of the example above is devoted to setting parameters. After defining all of the parameters, these are passed into a MPSOptimizerParameters object, and an MPSTrainingParameters object. An MPS is also initialised. In the penultimate line, the MPSOptimizer is initialised. Finally, the MPS is trained using the MPSOptimizer.train function.


Related Classes
-------

.. toctree::
   :maxdepth: 1

   sqMPSOptimizer
   SGDOptimizer
   MPSOptimizerParameters
   MPSTrainingParameters


Creating a new MPSOptimizer
-------------------

__init__(MPSNetwork, max_size, optional_parameters=MPSOptimizerParameters())
^^^^^^^^

 Initialises the optimiser.

 *MPSNetwork: MPS*
  The matrix product state network that will be optimised.
 *max_size: integer*
  The maximum size the tensors composing the MPS can grow to.
 *optional_parameters: MPSOptimizerParameters*
  Optional parameters for the MPSOptimizer. See documentation for MPSOptimizerParameters for more detail.

Training an MPS
----------------

train(self, data_source, batch_size, n_step, optional_parameters=MPSTrainingParameters())
^^^^^^^^

 Trains the network. If it is required to chain the training with other tensorflow steps, do not use this function. However, it may be helpful to base it on the way this function is implemented, as the way the MPS works is quite unique, so the way things will have to be done is somewhat different from how it is usually done in tensorflow. The trained weights are also saved to the file 'weights' at the end of each sweep.

 *data_source: (some subclass of) MPSDatasource*
  The data/labels that the MPS will be trained on.
 *batch_size: integer*
  The batch size to be used when feeding in data for the sweeps.
 *n_step: integer*
  The number of steps of training that should be performed. A step of training consists of a full sweep 'forward' and 'backward' such that the output leg is attached to a node at the same position as at the start. Typically, (if the batch size is all of the data), then a couple of steps should be enough to fully optimise the MPS.
 *optional_parameters: MPSTrainingParameters*
  Optional parameters for training in the MPSOptimizer. See documentation for MPSTrainingParameters for more detail.

train_step()
^^^^^^^^^
 A single step of training, as a tensorflow operation. Use this if you need to chain training with other tensorflow operations. If not, it is recommended to just use the train function.

 *returns: tensorflow scalar*
  The accuracy as calculated at the end of a training step.

