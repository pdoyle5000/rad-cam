import unittest
import numpy as np
from radcam.perturber import Perturber, Perturbation


class TestPerturber(unittest.TestCase):
    def setUp(self):
        self.single_chan_input_perfect = np.full((4, 4), 1)

    def test_perturb_perfect_size(self):
        ideal_black_output = np.array(
            [
                [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
            ]
        )
        output = Perturber(self.single_chan_input_perfect, filter_size=(2, 2)).perturb(
            Perturbation.black
        )
        np.testing.assert_array_equal(ideal_black_output, output)
