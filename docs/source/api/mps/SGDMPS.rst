SGDMPS
======

SGDMPS is a subclass of MPS used in place of the MPS for stochastic gradient descent optimization. A usage example is as follows:

.. code-block:: python

    import MNISTpreprocessing
    from trmps import *

    # Parameters
    input_size = 196
    shrink = True
    d_feature = 2
    d_output = 10
    permuted = False

    # Initialise with linear regression
    data_source = MNISTpreprocessing.MNISTDatasource(shrink, permuted = permuted)
    network = SimpleMPS(d_feature, d_output, input_size)
    network.prepare(data_source=data_source)
    feature, label = data_source.test_data
    network.test(feature, label)

Properties
--------

*d_feature: int (read only)*
 The sizes of the feature vectors. (Referred to as 'Local dimension' in the paper). Unlike with when MPSs are optimized via DMRG, the stochastic gradient descent optimization algorithm does not scale as (d_feature)^3 so it is not neccesary to be as careful with this.
*feature_reg: float (read only)*
 Parameter for regularising the weights. Set it larger than 1 for better performance.
*reg: float (read only)*
 The ratio between loss and the regularisation penalty. Set it smaller than 1 for better performance.
*start_node: tf.Tensor (read only)*
 Untrainable tf.Variable, the first node of the MPS. Is a vector. See `Notes for developement <https://github.com/TrMPS/MPS-MNIST/blob/master/Notes%20for%20developement.md>`_ for further details, but shouldn't need to interface with this directly.
*end_node: tf.Tensor (read only)*
 Untrainable tf.Variable, the last node of the MPS. Is a vector. See `Notes for developement <https://github.com/TrMPS/MPS-MNIST/blob/master/Notes%20for%20developement.md>`_ for further details, but shouldn't need to interface with this directly.
*nodes: list of tf.Variable*
 List of trainable tf.Variable, the middle nodes of the MPS.


Creating a new SGDMPS
-----------
__init__(d_feature, d_output, input_size, feature_reg=1.1, reg=0.001, special_node_loc=None):
^^^^^^^^
 Initialises the MPS. The prepare method must be called after this before  anything else can be done.

 *d_feature: integer*
  The sizes of the feature vectors. (Referred to as 'Local dimension' in the paper)
 *d_output: integer*
  The size of the output. e.g. with 10-class classification, expressed as a one-hot vector, this would be 10
 *input_size: int*
  The input size, i.e. the number of matrices composing the matrix product state
 *feature_reg: float*
  Parameter for regularising the weights. Set it larger than 1 for better performance.
 *reg: float*
  Ratio between loss and the regularisation penalty. Set it smaller than 1 for better performance.
