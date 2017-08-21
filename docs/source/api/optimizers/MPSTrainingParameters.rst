MPSTrainingParameters
============================

MPSTrainingParameters is used to pass in optional parameters for MPSOptimizer in the training step (the train method), and it helps keep the number of arguments for training MPSOptimizerParameters smaller.

Creating a new MPSTrainingParameters object
----------

__init__(rate_of_change=1000, initial_weights=None, _logging_enabled=False)
 *rate_of_change: float*
  The rate of change for the optimisation. Different values should be tried, as there is no 'right answer' that works for all situations, and depending on the data set, the same value can cause overshooting, or make the optimisation slower than it should be.
 *initial_weights: list*
  The initial weights for the network, if it is desired to override the default values from mps.prepare(self, data_source, iterations = 1000)
 *_logging_enabled: boolean*
  Whether certain things are logged to Tensorboard/ to a Chrome timeline.

Properties
----------
*rate_of_change: float*
 The rate of change for the optimisation. Different values should be tried, as there is no 'right answer' that works for all situations, and depending on the data set, the same value can cause overshooting, or make the optimisation slower than it should be.
*initial_weights: list*
 The initial weights for the network, if it is desired to override the default values from mps.prepare(self, data_source, iterations = 1000)
*_logging_enabled: boolean*
 Whether certain things are logged to Tensorboard/ to a Chrome timeline.
