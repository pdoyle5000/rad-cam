import base64
from enum import Enum, unique
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from typing_extensions import Protocol

from radcam.perturber import Perturbation, Perturber

IMAGE = Union[JpegImageFile, PngImageFile]
STD_DIM = 256


@unique
class Colors(Enum):
    green = "rgba(50, 255, 40,"
    orange = "rgba(255, 181, 43,"
    red = "rgba(255, 0, 0,"
    blue = "rgba(0, 166, 255,"
    yellow = "rgba(255, 255, 0,"
    magenta = "rgba(255, 0, 234,"


class ModelProtocol(Protocol):
    def predict(self, image: IMAGE) -> Union[list, np.ndarray, torch.Tensor]:
        ...


class RadCam:
    def __init__(
        self,
        model: ModelProtocol,
        filter_dims: Tuple[int, int] = (1, 1),
        image_width: int = STD_DIM,
        image_height: int = STD_DIM,
        tuple_index: int = None,
        color: Colors = Colors.orange,
    ):
        self.model = model
        self.image_width = image_width
        self.image_height = image_height
        self.filter_dims = filter_dims
        self.tuple_index = tuple_index
        self.color = color

    def heat_map(self, image: IMAGE, perturbation: Perturbation = Perturbation.black):
        image = _convert_type(image)
        perturber = Perturber(np.array(image), self.filter_dims)
        perturbed_array = perturber.perturb(perturbation)
        indices = perturber.block_locations
        actual_pred = self.model.predict(image)
        if self.tuple_index:
            actual_pred = actual_pred[self.tuple_index]
        actual_pred = _convert_type(actual_pred)
        preds = [self.model.predict(Image.fromarray(img)) for img in perturbed_array]
        if self.tuple_index:
            preds = [_convert_type(pred[self.tuple_index]) for pred in preds]
        preds = np.array(preds)
        diffs = calculate_diffs(actual_pred, preds)
        if len(diffs.shape) == 1:  # Torch tensors can get flattened.
            diffs = np.expand_dims(diffs, axis=1)
        return [
            go.Figure(self.generate_figure(image, indices, np.squeeze(probs)))
            for probs in diffs.T
        ]

    def generate_figure(self, image, verticies, heat_values):
        image = image.resize(
            (self.image_width, self.image_height)
        )  # Torch tensors can get flattened.
        data = self._generate_points(heat_values)
        return {
            "data": [data],
            "layout": {
                "shapes": self._generate_verticies(heat_values),
                "autosize": False,
                "hovermode": "closest",
                "margin": {"l": 20, "r": 20, "b": 20, "t": 20, "pad": 1},
                "xaxis": {
                    "range": [0, self.image_width],
                    "showgrid": False,
                    "zeroline": False,
                    "showline": False,
                    "ticks": "",
                    "showticklabels": True,
                },
                "yaxis": {
                    "range": [self.image_height, 0],
                    "showgrid": False,
                    "zeroline": False,
                    "showline": False,
                    "ticks": "",
                    "showticklabels": True,
                },
                "width": self.image_height + 40,
                "height": self.image_height + 40,
                "images": [
                    {
                        "source": image,
                        "xref": "x",
                        "yref": "y",
                        "x": 0,
                        "y": 0,
                        "sizex": self.image_width,
                        "sizey": self.image_height,
                        "sizing": "stretch",
                        "layer": "below",
                    }
                ],
            },
        }

    def _generate_verticies(self, heat):
        shapes = []
        width = self.filter_dims[0]
        height = self.filter_dims[1]
        pos_y = 0
        heat_pos = 0
        while pos_y + height <= self.image_height:
            pos_x = 0
            while pos_x + width <= self.image_width:
                intensity = np.around(heat[heat_pos], decimals=2)
                rect = {
                    "type": "rect",
                    "x0": pos_x,
                    "y0": pos_y,
                    "x1": pos_x + width,
                    "y1": pos_y + height,
                    "line_width": 0,
                    "fillcolor": f"{self.color.value} {intensity})",
                }
                shapes.append(rect)
                heat_pos += 1
                pos_x += width
            pos_y += height
        return shapes

    def _generate_points(self, heat):
        pos_y = 0
        width = self.filter_dims[0]
        height = self.filter_dims[1]
        x_data = []
        y_data = []
        while pos_y + height <= self.image_height:
            pos_x = 0
            while pos_x + width <= self.image_width:
                x_data.append(pos_x + (width) / 2)
                y_data.append(pos_y + (height) / 2)
                pos_x += width
            pos_y += height
        return {
            "x": x_data,
            "y": y_data,
            "mode": "markers",
            "text": heat,
            "hoverinfo": "text",
            "opacity": 0.05,
        }


def calculate_diffs(actual_preds, perturb_preds):
    return np.absolute(actual_preds - perturb_preds)


def _convert_type(
    preds: Union[torch.Tensor, pd.DataFrame, np.ndarray, list]
) -> np.ndarray:
    if isinstance(preds, torch.Tensor):
        return preds.numpy()
    if isinstance(preds, list):
        return np.array(preds)
    return preds
