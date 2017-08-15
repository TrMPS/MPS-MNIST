class MPSOptimizerParameters(object):
    def __init__(self, cutoff=1000,
                 reg=0.001, lr_reg=0.99, min_singular_value=10**(-4),
                 verbosity=0, armijo_coeff=0.5):
        self.cutoff = cutoff
        self.reg = reg
        self.lr_reg=lr_reg
        self.min_singular_value=min_singular_value
        self.verbosity=verbosity
        self.armijo_coeff=armijo_coeff

class MPSTrainingParameters(object):
    def __init__(self, rate_of_change=1000, initial_weights=None, _logging_enabled=False):
        self.rate_of_change = rate_of_change
        self.initial_weights = initial_weights
        self._logging_enabled = _logging_enabled
