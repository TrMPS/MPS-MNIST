shortMPSOptimizer
============================

shortMPSOptimizer is a subclass of MPSOptimizer for use with shortMPS to train models that can be used for generation. The key difference is that it restricts the sweeping to a certain range given when an instance is created. It will be deprecated in the near future and its functionality rolled into MPSOptimizer.


Creating a new shortMPSOptimizer
-------------------

__init__(MPSNetwork, max_size, optional_parameters=MPSOptimizerParameters())
^^^^^^^^

 Initialises the optimiser.

 *MPSNetwork: MPS, sqMPS or shortMPS*
  The matrix product state that will be optimised. Take note that if an sqMPS is passed in, the MPSOptimizer cannot use the Hessian when optimizing. Despite being a subclass of MPS, passing in an SGDMPS is not supported. If you want to optimize an SGDMPS, use SGDOptimizer. Note that when an MPS or sqMPS are passed in and sweep_range is None, a shortMPSOptimizer will act in the same way as an MPSOptimizer.
 *max_size: integer*
  The maximum size the tensors composing the MPS can grow to.
 *sweep_range: tuple (integer, integer)*
  The first and last indices between which the MPS will be optimized.
 *optional_parameters: MPSOptimizerParameters*
  Optional parameters for the MPSOptimizer. See documentation for MPSOptimizerParameters for more detail.
