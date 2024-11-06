# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module contains utility functions for object detection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:

    import torch


def convert_tf_to_pt(y: list[dict[str, np.ndarray]], height: int, width: int) -> list[dict[str, np.ndarray]]:
    """
    :param y: Target values of format `list[dict[Tensor]]`, one for each input image. The fields of the dict are as
              follows:

              - boxes (FloatTensor[N, 4]): the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and
                                           0 <= y1 < y2 <= H in scale [0, 1].
              - labels (Int64Tensor[N]): the labels for each image
              - scores (Tensor[N]): the scores or each prediction.
    :param height: Height of images in pixels.
    :param width: Width if images in pixels.

    :return: Target values of format `list[dict[Tensor]]`, one for each input image. The fields of the dict are as
             follows:

             - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                          0 <= y1 < y2 <= H in image scale.
             - labels (Int64Tensor[N]): the labels for each image
             - scores (Tensor[N]): the scores or each prediction.
    """
    for i, _ in enumerate(y):
        y[i]["boxes"][:, 0] *= height
        y[i]["boxes"][:, 1] *= width
        y[i]["boxes"][:, 2] *= height
        y[i]["boxes"][:, 3] *= width

        y[i]["boxes"] = y[i]["boxes"][:, [1, 0, 3, 2]]

        y[i]["labels"] = y[i]["labels"] + 1

    return y


def convert_pt_to_tf(y: list[dict[str, np.ndarray]], height: int, width: int) -> list[dict[str, np.ndarray]]:
    """
    :param y: Target values of format `list[dict[Tensor]]`, one for each input image. The fields of the dict are as
              follows:

              - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                               0 <= y1 < y2 <= H in image scale.
              - labels (Int64Tensor[N]): the labels for each image
              - scores (Tensor[N]): the scores or each prediction.
    :param height: Height of images in pixels.
    :param width: Width if images in pixels.

    :return: Target values of format `list[dict[Tensor]]`, one for each input image. The fields of the dict are as
             follows:

             - boxes (FloatTensor[N, 4]): the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and
                                          0 <= y1 < y2 <= H in scale [0, 1].
             - labels (Int64Tensor[N]): the labels for each image
             - scores (Tensor[N]): the scores or each prediction.
    """
    for i, _ in enumerate(y):

        y[i]["boxes"][:, 0] /= width
        y[i]["boxes"][:, 1] /= height
        y[i]["boxes"][:, 2] /= width
        y[i]["boxes"][:, 3] /= height

        y[i]["boxes"] = y[i]["boxes"][:, [1, 0, 3, 2]]

        y[i]["labels"] = y[i]["labels"] - 1

    return y


def cast_inputs_to_pt(
    x: np.ndarray | "torch.Tensor",
    y: list[dict[str, np.ndarray | "torch.Tensor"]] | None = None,
) -> tuple["torch.Tensor", list[dict[str, "torch.Tensor"]] | None]:
    """
    Cast object detection inputs `(x, y)` to PyTorch tensors.

    :param x: Samples of shape NCHW or NHWC.
    :param y: Target values of format `list[dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                The fields of the dict are as follows:

                - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                - labels [N]: the labels for each image.
    :return: Object detection inputs `(x, y)` as tensors.
    """
    import torch

    # Convert images into tensor
    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x)
    else:
        x_tensor = x

    y_tensor: list[dict[str, torch.Tensor]] | None = None

    # Convert labels into tensor
    if isinstance(y, list):
        y_tensor = []
        for y_i in y:
            y_t = {}

            if isinstance(y_i["boxes"], np.ndarray):
                y_t["boxes"] = torch.from_numpy(y_i["boxes"]).to(dtype=torch.float32)
            else:
                y_t["boxes"] = y_i["boxes"]

            if isinstance(y_i["labels"], np.ndarray):
                y_t["labels"] = torch.from_numpy(y_i["labels"]).to(dtype=torch.int64)
            else:
                y_t["labels"] = y_i["labels"]

            if "masks" in y_i:
                if isinstance(y_i["masks"], np.ndarray):
                    y_t["masks"] = torch.from_numpy(y_i["masks"]).to(dtype=torch.uint8)
                else:
                    y_t["masks"] = y_i["masks"]

            y_tensor.append(y_t)
    elif isinstance(y, dict):
        y_tensor = []
        for i in range(len(y["boxes"])):
            y_t = {}

            y_t["boxes"] = y["boxes"][i]
            y_t["labels"] = y["labels"][i]
            if "masks" in y:
                y_t["masks"] = y["masks"][i]

            y_tensor.append(y_t)
    else:
        y_tensor = y

    return x_tensor, y_tensor
