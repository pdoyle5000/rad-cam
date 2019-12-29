import unittest
from typing import List
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from radcam.radcam import RadCam


class MockModel:
    def predict(self, image: JpegImageFile) -> List[float]:
        return [0.8]


class TestRadCam(unittest.TestCase):
    def setUp(self):
        self.input_image = Image.fromarray(np.zeros((4, 4))).convert("L")

        self.shape_corners = [
            {"x0": 0, "x1": 2, "y0": 0, "y1": 2},
            {"x0": 2, "x1": 4, "y0": 0, "y1": 2},
            {"x0": 0, "x1": 2, "y0": 2, "y1": 4},
            {"x0": 2, "x1": 4, "y0": 2, "y1": 4},
        ]

    def testOutputValid(self):
        model = MockModel()
        radcam = RadCam(
            model,
            image_width=self.input_image.size[0],
            image_height=self.input_image.size[1],
            filter_dims=(2, 2),
        )
        heatmap = radcam.heat_map(self.input_image)[0]
        self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
        self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
        np.testing.assert_array_equal(heatmap.data[0].text, [0.8, 0.8, 0.8, 0.8])
        for i, shape in enumerate(heatmap.layout.shapes):
            self.assertEqual(shape.x0, self.shape_corners[i]["x0"])
            self.assertEqual(shape.x1, self.shape_corners[i]["x1"])
            self.assertEqual(shape.y0, self.shape_corners[i]["y0"])
            self.assertEqual(shape.y1, self.shape_corners[i]["y1"])
