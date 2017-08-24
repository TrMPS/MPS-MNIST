SGDOptimizer
============================

SGDOptimizer is used to optimise SGDMPS using stochastic gradient descent. S gradient descent is implemented using tensorflow AdamOptimizer. Despite having a similar API, SGDOptimizer is not related to MPSOptimizer or SingleSiteMPSOptimizer (While they inherit from a common base class, SGDOptimizer does not. A usage example is as below:

.. code-block:: python

    from trmps import *
    from MNISTpreprocessing import MNISTDatasource

    # Parameters
    d_feature = 2
    d_output = 10
    batch_size = 100
    permuted = False
    shuffled = True
    shrink = True
    input_size = 196

    rate_of_change = 0.001
    feature_reg=1.1
    reg=0.1/batch_size
    n_step = 1200

    data_source = MNISTDatasource(shrink=shrink, permuted=permuted, shuffled=shuffled)

    # Make the SGDMPS network and initialise with linear regression
    network = SGDMPS(d_feature, d_output, input_size,
                        feature_reg=feature_reg,
                        reg=reg)
    network.prepare(data_source)

    # Optimise with stochastic gradient descent
    optimizer = SGDOptimizer(network)
    optimizer.train(data_source, batch_size, n_step,
                    rate_of_change=rate_of_change)

Creating a new SGDOptimizer
---------------

__init__(self, network)
^^^^^^^^

 *MPSNetwork: SGDMPS*
  The SGDMPS to be optimised.

Training an SGDMPS
----------------

train(data_source, batch_size, n_steps, rate_of_change=0.001)
^^^^^^^^^^
 Trains the SGDMPS. The weights are saved under weights_sgd as a pickle file at the end of the training.

 *data_source: (some subclass of) MPSDatasource*
  The data/labels that the MPS will be trained on.
 *batch_size: integer*
  The batch size used for one step of stochastic gradient descent.
 *n_steps: integer*
  Number of batches to pass to the MPS.
 *rate_of_change: float*
  Rate of change to pass into the AdamOptimizer

