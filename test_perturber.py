import unittest
import numpy as np
from radcam.perturber import Perturber, Perturbation


class TestPerturber(unittest.TestCase):
    def setUp(self):
        self.single_chan_input_perfect = np.full((4, 4), 1)
        self.single_chan_input_imperfect = np.full((5, 5), 1)
        self.rgb_input_perfect = np.full((4, 4, 3), 1)

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

    def test_perturb_imperfect_size(self):
        ideal_black_output = np.array(
            [
                [
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
            ]
        )
        output = Perturber(
            self.single_chan_input_imperfect, filter_size=(2, 2)
        ).perturb(Perturbation.black)
        np.testing.assert_array_equal(ideal_black_output, output)

    def test_perturb_rgb_perfect_size(self):
        ideal_black_output = np.array(
            [
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
                ],
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
                ],
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
                ],
            ]
        )
        output = Perturber(self.rgb_input_perfect, filter_size=(2, 2)).perturb(
            Perturbation.black
        )
        print("OUTPUT")
        print(output.shape)
        output = np.transpose(output, (3, 0, 1, 2))
        np.testing.assert_array_equal(ideal_black_output, output)
