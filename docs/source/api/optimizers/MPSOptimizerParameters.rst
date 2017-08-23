MPSOptimizerParameters
============================

MPSOptimizerParameters is used to pass in optional parameters for MPSOptimizer, and it helps keep the number of arguments for initialising MPSOptimizerParameters smaller.

Creating a new MPSOptimizerParameters object
-----------

__init__(cutoff=1000, reg=0.001, lr_reg=0.99, min_singular_value=10**(-4), verbosity=0, armijo_coeff=10**(-4), use_hessian=False, armijo_iterations=10)
^^^^^^^^^^^^

 *cutoff: float*
  The cutoff value for the gradient. Anything above this is clipped off. Deprecated and does not do anything. Will be removed in a future release.
 *reg: float*
  The amount of regularisation to include in the cost function. The higher this value is, the more the regularization of the weight matrices matters.
 *lr_reg: float, should be between 0 and 1*
  lr_reg is the learning rate regularisation, and this value determines how much the learning rate decreases by as more training is done. A value of 0 keeps the learning rate constant. When the value is greater than 0, instead of the learning rate getting smaller than the provided learning rate as more steps are done, the learning rate is boosted at the start and as the number of steps goes on, approaches the provided learning rate.
 *min_singular_value: float*
  Below this value, values in the decomposed singular values are ignored. Set this value high if you want a compact model, and low if you want a more accurate model.
 *verbosity: integer*
  This value controls how much the optimizer prints out during training. Set it to 0 to not have anything printed out, and a positive number n to have the first n loops printed out. Set this to a negative value to have the optimizer print out logging information every time it loops.
 *armijo_coeff: float*
  The coefficient for the tangential part of the armijo equation. The higher this value is, the more stringent the armijo condition is.
 *use_hessian: bool*
  Controls whether to use the Hessian or not in calculating the gradient

Properties
----------
Although these properties can be changed at any point, once the MPSOptimizer is initialised using the MPSOptimizerParameters object, changing these properties should make no difference.

*cutoff: float*
 The cutoff value for the gradient. Anything above this is clipped off. Deprecated and does not do anything. Will be removed in a future release.
*reg: float*
 The amount of regularisation to include in the cost function. The higher this value is, the more the regularization of the weight matrices matters.
*lr_reg: float, should be between 0 and 1*
 lr_reg is the learning rate regularisation, and this value determines how much the learning rate decreases by as more training is done. A value of 0 keeps the learning rate constant. When the value is greater than 0, instead of the learning rate getting smaller than the provided learning rate as more steps are done, the learning rate is boosted at the start and as the number of steps goes on, approaches the provided learning rate.
*min_singular_value: float*
 Below this value, values in the decomposed singular values are ignored. Set this value high if you want a compact model, and low if you want a more accurate model.
*verbosity: integer*
 This value controls how much the optimizer prints out during training. Set it to 0 to not have anything printed out, and a positive number n to have the first n loops printed out. Set this to a negative value to have the optimizer print out logging information every time it loops.
*armijo_coeff: float*
 The coefficient for the tangential part of the armijo equation. The higher this value is, the more stringent the armijo condition is.
*use_hessian: bool*
 Controls whether to use the Hessian or not in calculating the gradient
