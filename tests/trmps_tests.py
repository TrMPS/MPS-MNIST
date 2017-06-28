import unittest
import sys
sys.path.append('..')

import trmps.mps as mps

class MPSTest(unittest.TestCase):
    def test_correct_dimensions_of_phi(self):
        self.state = mps.MPS(2, 2, 10, 10, 10)
        self.state.prepare()
        self.assertEqual(self.state.phi.shape.dims[0].value, 10)
