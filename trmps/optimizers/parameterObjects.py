class MPSOptimizerParameters(object):
    """
    MPSOptimizerParameters is used to pass in optional parameters for MPSOptimizer,
    as it saves having a large number of optional parameters.
    """
    def __init__(self, cutoff=1000,
                 reg=0.001, lr_reg=0.99, min_singular_value=10**(-4),
                 verbosity=0, armijo_coeff=10**(-4), use_hessian=False,
                 armijo_iterations=10, path="MPSconfig", updates_per_step=1):
        """
        :param cutoff: float
            The cutoff value for the gradient. Anything above this is clipped off.
            Deprecated.
        :param reg: float
            The amount of regularisation to include in the cost function.
            The higher this value is, the more the regularization of the weight matrices matters.
        :param lr_reg: float, should be between 0 and 1
            lr_reg is the learning rate regularisation, and this value determines how much the
            learning rate decreases by as more training is done.
            A value of 0 keeps the learning rate constant.
            When the value is greater than 0, instead of the learning rate getting smaller than the
            provided learning rate as more steps are done, the learning rate is boosted at the start
            and as the number of steps goes on, approaches the provided learning rate.
        :param min_singular_value: float
            Below this, values in the decomposed singular values are ignored.
            Set this value high if you want a compact model,
            and low if you want a more accurate model.
        :param verbosity: integer
            This value controls how much the optimizer prints out during training.
            Set it to 0 to not have anything printed out, and a positive number n to have
            the first n loops printed out.
            Set this to a negative value to have the optimizer print out logging information
            every time it loops.
        :param armijo_coeff: float
            The coefficient for the tangential part of the armijo equation.
            The higher this value is, the more stringent the armijo condition is.
        :param use_hessian: bool
            Controls whether to use the Hessian or not in calculating the gradient
        """
        self.cutoff = cutoff
        self.reg = reg
        self.lr_reg = lr_reg
        self.min_singular_value = min_singular_value
        self.verbosity = verbosity
        self.armijo_coeff = armijo_coeff
        self.use_hessian = use_hessian
        self.armijo_iterations = armijo_iterations
        self.path = path
        self.updates_per_step = updates_per_step

class MPSTrainingParameters(object):
    """
    MPSTrainingParameters is used to pass in optional parameters for MPSOptimizer in the training step,
    as it saves having a large number of optional parameters.
    """
    def __init__(self, rate_of_change=1000, initial_weights=None, verbose_save=True,
                 _logging_enabled=False,):
        """
        :param rate_of_change: float
            The rate of change for the optimisation.
            Different values should be tried, as there is no 'right answer' that works for
            all situations, and depending on the data set, the same value can cause
            overshooting, or make the optimisation slower than it should be.
        :param initial_weights: list
            The initial weights for the network, if it is desired to override the default values
            from mps.prepare(self, data_source, iterations = 1000).
            Deprecated.
        :param _logging_enabled: boolean
            Whether certain things are logged to Tensorboard/ to a Chrome timeline.
        """
        self.rate_of_change = rate_of_change
        self.initial_weights = initial_weights
        self._logging_enabled = _logging_enabled
        self.verbose_save = verbose_save
