from scipy.ndimage.filters import gaussian_filter
import numpy as np
from typing import Tuple
from enum import Enum, unique
from copy import deepcopy


@unique
class Perturbation(Enum):
    black = 1
    white = 2
    noise = 3
    gaussian = 4
    median = 5
    mean = 6


class Perturber:
    def __init__(self, input_array: np.array, filter_size: Tuple[int, int]):
        self.input_array = deepcopy(input_array)
        self.filter_size = filter_size
        self.block_locations = self._get_blocks()

    def perturb(self, perturbation_type: Perturbation = Perturbation.black) -> np.array:
        if perturbation_type == Perturbation.gaussian:
            return self._apply_gaussian()
        perturbation_map = {
                Perturbation.black: self._apply_black,
                Perturbation.white: self._apply_white,
                Perturbation.noise: self._apply_noise,
                Perturbation.median: self._apply_median,
                Perturbation.mean: self._apply_mean}
        return perturbation_map[perturbation_type]()

    def _apply_black(self):
        return self._apply_simple_filter(
                np.zeros(self.filter_size, dtype=int))

    def _apply_white(self):
        return self._apply_simple_filter(
                np.full(self.filter_size, 255, dtype=int))

    def _apply_noise(self):
        return self._apply_simple_filter(
                np.random.randint(256, size=self.filter_size))

    def _apply_mean(self):
        mean = np.mean(self.input_array)
        return self._apply_simple_filter(
                np.full(self.filter_size, mean, dtype=int))

    def _apply_median(self):
        median = np.median(self.input_array)
        return self._apply_simple_filter(
                np.full(self.filter_size, median, dtype=int))

    def _apply_gaussian(self):
        # cant just apply a filter.
        # for each block in the filter, create a new gaussian block
        # return gaussian_filter(array)
        return self._apply_noise  # todo.

    def _apply_simple_filter(self, array: np.ndarray):
        output = []
        for block in self._get_blocks():
            perturbed_array = deepcopy(self.input_array)
            for (x, y), _ in np.ndenumerate(array):
                perturbed_array[x + block[0], y + block[1]] = array[x, y]
            output.append(perturbed_array)
        return np.array(output)

    def _get_blocks(self):
        block_locs = []
        for x in range(0, self.input_array.shape[1] - 1, self.filter_size[0]):
            for y in range(0, self.input_array.shape[0] - 1, self.filter_size[1]):
                block_locs.append((x, y))
        self.block_locations = block_locs
        return block_locs
