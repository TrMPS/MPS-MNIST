shortMPS
========

shortMPS is a subclass of MPS that has a slightly different structure, which is required for generation. Namely, the start_node and end_node are of a different shape. Certain matrices (usually the first and last ones) should not be trained, and to do this, the shortMPSOptimizer class should be used instead of MPSOptimizer or sqMPSOptimizer.
