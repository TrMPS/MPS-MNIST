SingleSiteMPSOptimizer
============================

SingleSiteMPSoptimizer has the exact same API as MPSOptimizer, but uses a single-site DMRG method for optimization as opposed to the two-site one in MPSOptimizer. It runs much faster than MPSOptimizer (about an O(local dimension) faster), and produces results that are roughly as good as an MPSOptimizer.
