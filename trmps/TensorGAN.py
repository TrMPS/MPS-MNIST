from mps import MPS
import numpy as np
import tensorflow as tensorflow

class GeneratorMPS(MPS):
    def create(self, seed):
        return super().predict(seed)
