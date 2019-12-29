import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
from typing import Union, List, Tuple
from typing_extensions import Protocol
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from PIL import Image

from radcam.perturber import Perturber, Perturbation


IMAGE = Union[JpegImageFile, PngImageFile]
STD_DIM = 256


class ModelProtocol(Protocol):
    def predict(self, image: IMAGE) -> List[float]:
        ...


class RadCam:
    def __init__(
        self,
        model: ModelProtocol,
        filter_dims: Tuple[int, int] = (1, 1),
        image_width: int = STD_DIM,
        image_height: int = STD_DIM,
    ):
        self.model = model
        self.image_width = image_width
        self.image_height = image_height
        self.filter_dims = filter_dims

    # Only accounting for single channel images.
    def heat_map(self, image: IMAGE):
        perturber = Perturber(np.array(image), self.filter_dims)
        perturbed_array = perturber.perturb(Perturbation.black)
        indices = perturber.block_locations
        preds = np.array(
            [self.model.predict(Image.fromarray(img)) for img in perturbed_array]
        )
        preds_by_class = preds.T
        return [
            go.Figure(self.generate_figure(image, indices, probs))
            for probs in preds_by_class
        ]

    def generate_figure(self, image, verticies, heat_values):
        image = image.resize((self.image_width, self.image_height))
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
                rect = {
                    "type": "rect",
                    "x0": pos_x,
                    "y0": pos_y,
                    "x1": pos_x + width,
                    "y1": pos_y + height,
                    "line": {"color": "rgba(50, 255, 40, .4)"},
                }
                rect["fillcolor"] = "rgba(50, 255, 40, {})".format(
                    np.around(heat[heat_pos], decimals=2)
                )
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
