import unittest
import torch
from typing import List, Tuple
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from radcam.radcam import RadCam, calculate_diffs


class MockModel:
    def predict(self, image: JpegImageFile) -> List[float]:
        return [0.8]


class MockModelMultiClass:
    def predict(self, image: JpegImageFile) -> List[float]:
        return [0.9, 0.2, 0.1]


class MockNumpyOutputModel:
    def predict(self, image: JpegImageFile) -> np.ndarray:
        return np.array([0.8])


class MockTorchOutputModel:
    def predict(self, image: JpegImageFile) -> torch.Tensor:
        return torch.tensor([0.8])


class MockTupleOutputModel:
    def predict(self, image: JpegImageFile) -> Tuple[str, torch.Tensor]:
        return "someclass", torch.tensor([0.9, 0.2, 0.1])


class TestRadCam(unittest.TestCase):
    def setUp(self):
        self.input_image = Image.fromarray(np.zeros((4, 4))).convert("L")
        self.input_rgb_image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))

        self.shape_corners = [
            {"x0": 0, "x1": 2, "y0": 0, "y1": 2},
            {"x0": 2, "x1": 4, "y0": 0, "y1": 2},
            {"x0": 0, "x1": 2, "y0": 2, "y1": 4},
            {"x0": 2, "x1": 4, "y0": 2, "y1": 4},
        ]

    def test_calculate_diffs_binary_pred(self):
        pred = np.array([1.0])
        altered_pred = np.array([[0.4], [0.3], [0.2], [2.1]])
        output = np.array([[0.6], [0.7], [0.8], [1.1]])
        np.testing.assert_array_almost_equal(
            calculate_diffs(pred, altered_pred), output, decimal=10
        )

    def test_calculate_diffs_multiple_pred(self):
        pred = np.array([0.9, 0.3])
        altered_pred = np.array([[0.4, 0.5], [0.3, 0.3]])
        output = np.array([[0.5, 0.2], [0.6, 0.0]])
        np.testing.assert_array_almost_equal(
            calculate_diffs(pred, altered_pred), output, decimal=10
        )

    def test_single_chan_output_valid(self):
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
        np.testing.assert_array_equal(heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0])
        for i, shape in enumerate(heatmap.layout.shapes):
            self.assertEqual(shape.x0, self.shape_corners[i]["x0"])
            self.assertEqual(shape.x1, self.shape_corners[i]["x1"])
            self.assertEqual(shape.y0, self.shape_corners[i]["y0"])
            self.assertEqual(shape.y1, self.shape_corners[i]["y1"])

    def test_rgb_output_valid(self):
        model = MockModel()
        radcam = RadCam(
            model,
            image_width=self.input_rgb_image.size[0],
            image_height=self.input_rgb_image.size[1],
            filter_dims=(2, 2),
        )
        heatmap = radcam.heat_map(self.input_rgb_image)[0]
        self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
        self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
        np.testing.assert_array_equal(heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0])
        for i, shape in enumerate(heatmap.layout.shapes):
            self.assertEqual(shape.x0, self.shape_corners[i]["x0"])
            self.assertEqual(shape.x1, self.shape_corners[i]["x1"])
            self.assertEqual(shape.y0, self.shape_corners[i]["y0"])
            self.assertEqual(shape.y1, self.shape_corners[i]["y1"])

    def test_single_chan_multi_class(self):
        model = MockModelMultiClass()
        radcam = RadCam(
            model,
            image_width=self.input_image.size[0],
            image_height=self.input_image.size[1],
            filter_dims=(2, 2),
        )
        heatmaps = radcam.heat_map(self.input_image)
        self.assertEqual(len(heatmaps), 3)
        for heatmap in heatmaps:
            self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
            self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
            np.testing.assert_array_equal(heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0])

    def test_rgb_multi_class(self):
        model = MockModelMultiClass()
        radcam = RadCam(
            model,
            image_width=self.input_rgb_image.size[0],
            image_height=self.input_rgb_image.size[1],
            filter_dims=(2, 2),
        )
        heatmaps = radcam.heat_map(self.input_rgb_image)
        self.assertEqual(len(heatmaps), 3)
        for heatmap in heatmaps:
            self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
            self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
            np.testing.assert_array_equal(heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0])

    def test_conversion_of_different_pred_types_single(self):
        for model in [MockNumpyOutputModel(), MockTorchOutputModel()]:
            radcam = RadCam(
                model,
                image_width=self.input_rgb_image.size[0],
                image_height=self.input_rgb_image.size[1],
                filter_dims=(2, 2),
            )
            heatmaps = radcam.heat_map(self.input_rgb_image)
            for heatmap in heatmaps:
                self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
                self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
                np.testing.assert_array_equal(
                    heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0]
                )

    def test_tuple_putput_ok(self):
        model = MockTupleOutputModel()
        radcam = RadCam(
            model,
            image_width=self.input_image.size[0],
            image_height=self.input_image.size[1],
            filter_dims=(2, 2),
            tuple_index=1,
        )
        heatmap = radcam.heat_map(self.input_image)[0]
        self.assertEqual((1.0, 3.0, 1.0, 3.0), heatmap.data[0].x)
        self.assertEqual((1.0, 1.0, 3.0, 3.0), heatmap.data[0].y)
        np.testing.assert_array_equal(heatmap.data[0].text, [0.0, 0.0, 0.0, 0.0])
        for i, shape in enumerate(heatmap.layout.shapes):
            self.assertEqual(shape.x0, self.shape_corners[i]["x0"])
            self.assertEqual(shape.x1, self.shape_corners[i]["x1"])
            self.assertEqual(shape.y0, self.shape_corners[i]["y0"])
            self.assertEqual(shape.y1, self.shape_corners[i]["y1"])
