from typing import List

import numpy as np
import torch


def non_max_suppression(
    prediction: List[np.ndarray], iou_thres: float = 0.45, class_agnostic=True
) -> List[np.ndarray]:
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    output = []
    for x in prediction:
        i = _nms(x[:, :5], iou_thres)
        output.append(x[i])
    return output


def _nms(box_scores: np.ndarray, iou_thres: float = 0.45) -> List[np.ndarray]:
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]

    picked = []
    indexes = np.argsort(scores)[::-1]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = _box_iou(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_thres]
    return picked


def _box_area(left_top: np.ndarray, right_bottom: np.ndarray):
    """Compute the areas of rectangles given two corners."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b8:5668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L89-L100
    width_height = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return width_height[..., 0] * width_height[..., 1]


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Return intersection-over-union (Jaccard index) of boxes."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L103-L119
    overlap_left_top = np.maximum(boxes1[..., :2], boxes2[..., :2])
    overlap_right_bottom = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    overlap_area = _box_area(overlap_left_top, overlap_right_bottom)
    area1 = _box_area(boxes1[..., :2], boxes1[..., 2:])
    area2 = _box_area(boxes2[..., :2], boxes2[..., 2:])
    return overlap_area / (area1 + area2 - overlap_area + eps)
