import numpy as np
from typing import Tuple
from enum import Enum, unique
from copy import deepcopy


@unique
class Perturbation(Enum):
    black = 1
    white = 2
    noise = 3


# TODOS: filters for white and random noise
# account for imperfect image size to filter sizes.
# account for more than one channel.


class Perturber:
    def __init__(self, input_array: np.array, filter_size: Tuple[int, int]):
        self.input_array = deepcopy(input_array)
        self.filter_size = filter_size
        self.block_locations = self._get_blocks()

    def perturb(self, perturbation_type: Perturbation) -> np.array:
        perturbation_map = {Perturbation.black: self._apply_black}
        return perturbation_map[perturbation_type]()

    def _apply_black(self):
        black = np.zeros(self.filter_size, dtype=int)
        output = []
        for block in self._get_blocks():
            perturbed_array = deepcopy(self.input_array)
            for (x, y), _ in np.ndenumerate(black):
                perturbed_array[x + block[0], y + block[1]] = black[x, y]
            output.append(perturbed_array)
        return np.array(output)

    def _get_blocks(self):
        block_locs = []
        for x in range(0, self.input_array.shape[1] - 1, self.filter_size[0]):
            for y in range(0, self.input_array.shape[0] - 1, self.filter_size[1]):
                block_locs.append((x, y))
        self.block_locations = block_locs
        return block_locs
