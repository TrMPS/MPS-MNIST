Generating samples
==========

Introduction
--------------------------

A key advantage of Matrix Product States (MPS) lies in the fact that the same model used to generate samples. TrMPS implements this functionality, but currently in a highly limited and experimental (even more so than the rest of the library) fashion. There is currently just one script, generation/generator.py, that contains the MPSGenerator class. It is highly specialised to the MNIST dataset, but will (hopefully) be expanded in the future. An example of the current usage of the class can be found at the bottom of the generator.py file, and the current (31st March 2018) API is documented below but may change in the future as the class is more generalised.

Creating a new MPSGenerator
-----------
__init__(MPSNetwork)
^^^^^^^^
 Initialises the MPSGenerator.

 *MPSNetwork: shortMPS*
  The network from which samples will be generated. A shortMPS must be passed in due to the fact that the MPS must be in a certain form for the generating algorithm to work efficiently.

Generating samples
---------
generate(n_samples, digit)
^^^^^^^^
 A tensorflow operation that generates the desired number of samples. As this is a tensorflow operation (unlike the train method on MPSOptimizer), it must be run using a tensorflow session at some point later. Currently tailored to the MNIST dataset.

 *n_samples: integer*
  The number of samples that you want to generate.

 *digit: integer*
  The digit for which you want to generate samples.

 *returns: a tuple of two tensorflow arrays*
  The first element of the tuple are the samples with the dimensions (length of MPS, batch size), and the second element represents the values of the pdfs for each sample.

