from mps import MPS
import numpy as np
import tensorflow as tensorflow

class GeneratorMPS(MPS):
    def create(seed):
        return super().predict(seed)
