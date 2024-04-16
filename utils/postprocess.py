from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from utils.info import *
from utils.postprocess_func.output_decoder import (
    InsSegDecoder,
    ObjDetDecoder,
    PoseEstDecoder,
)


def getPostProcesser(app_type, model_name, model_cfg, class_names, use_tracking=True):
    assert app_type in [
        "object_detection",
        "pose_estimation",
        "instance_segmentation",
    ], "sdfasd"
    if app_type == "object_detection":
        postproc = ObjDetPostprocess(model_name, model_cfg, class_names, use_tracking)
    elif app_type == "pose_estimation":
        postproc = PoseEstPostprocess(model_name, model_cfg, class_names, use_tracking)
    elif app_type == "instance_segmentation":
        postproc = InsSegPostProcess(model_name, model_cfg, class_names, use_tracking)
    return postproc


# Postprocess for Object Detection
class ObjDetPostprocess:
    def __init__(
        self, model_name: str, model_cfg, class_names, use_traking: bool = True
    ):
        model_cfg.update({"use_tracker": use_traking})
        self.postprocess_func = ObjDetDecoder(model_name, **model_cfg)
        self.class_names = class_names

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        ## Consider batch 1

        predictions = self.postprocess_func(outputs, contexts, img.shape[:2])
        assert len(predictions) == 1, f"{len(predictions)}!=1"

        predictions = predictions[0]
        num_prediction = predictions.shape[0]

        if num_prediction == 0:
            return img

        bboxed_img = draw_bbox(img, predictions, self.class_names)
        return bboxed_img


# Postprocess for Pose Estimation
class PoseEstPostprocess:
    def __init__(
        self, model_name: str, model_cfg, class_names, use_traking: bool = True
    ):
        model_cfg.update({"use_tracker": use_traking})
        self.postprocess_func = PoseEstDecoder(model_name, **model_cfg)
        self.class_names = class_names
        self.s_idx = 5 if "yolov8" in model_name else 6

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        ## Consider batch 1
        predictions = self.postprocess_func(outputs, contexts, img.shape[:2])
        assert len(predictions) == 1, f"{len(predictions)}!=1"

        predictions = predictions[0]
        num_prediction = predictions.shape[0]

        if num_prediction == 0:
            return img

        predictions = predictions[:, self.s_idx :]
        pose_img = draw_pose(img, predictions)
        return pose_img


# Postprocess for Instance Segmentation
class InsSegPostProcess:
    def __init__(
        self, model_name: str, model_cfg, class_names, use_traking: bool = True
    ):
        model_cfg.update({"use_tracker": use_traking})
        self.postprocess_func = InsSegDecoder(model_name, **model_cfg)
        self.class_names = class_names

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        ## Consider batch 1

        predictions = self.postprocess_func(outputs, contexts, img.shape[:2])
        assert len(predictions) == 1, f"{len(predictions)}!=1"

        predictions = predictions[0]
        bbox, ins_mask = predictions
        num_prediction = bbox.shape[0]

        if num_prediction == 0 or ins_mask is None:
            return img

        ins_mask_img = draw_instance_mask(img, ins_mask, bbox, self.class_names)
        return ins_mask_img


# Draw Output on an Original Image
def draw_bbox(
    img: np.ndarray, predictions: np.ndarray, class_names: List[str]
) -> np.ndarray:
    ## Draw box on org image
    for prediction in predictions:
        mbox = [int(i) for i in prediction[:4]]
        score = prediction[4]
        class_id = int(prediction[5])

        color = COLORS[class_id % len(COLORS)]
        if len(prediction) != 6:
            tracking_id = int(prediction[-1]) % 30
            color = COLORS[tracking_id % len(COLORS)]

        label = f"{class_names[class_id]} {int(score*100)}%"
        img = plot_onx_box(mbox, img, color, label)

    return img


def plot_onx_box(
    box: List[int],
    img: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
    line_thickness: int = None,
) -> np.ndarray:
    tl = line_thickness or round(0.0005 * max(img.shape[0:2])) + 1
    tf = max(tl - 1, 1)
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(
        img,
        label,
        (c1[0], c1[1] - 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def draw_instance_mask(
    img: np.ndarray,
    masks: np.ndarray,
    bbox: np.ndarray,
    class_names: List[str],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    ## Draw instance mask
    for box, mask in zip(bbox, masks):
        class_id = int(box[5])
        color = COLORS[class_id % len(COLORS)]
        if len(box) != 6:
            tracking_id = int(box[-1])
            color = COLORS[tracking_id % len(COLORS)]

        mask = mask.astype(bool)
        img[:, :, :][mask] = img[:, :, :][mask] * beta + np.array(color) * alpha
    return img


def draw_pose(
    img: np.ndarray, predictions: np.ndarray, line_thickness: int = None
) -> np.ndarray:
    ## Draw Pose

    tl = line_thickness or round(0.0005 * max(img.shape[0:2])) + 1
    kpt_color = [62, 0, 198]

    for i, prediction in enumerate(predictions):

        for idx in range(len(SKELETONS)):
            color = POSE_LIMB_COLOR[idx]
            skeleton = SKELETONS[idx]

            (pos_x_1, pos_y_1, _) = prediction[
                (skeleton[0] - 1) * 3 : (skeleton[0] - 1) * 3 + 3
            ]
            (pos_x_2, pos_y_2, _) = prediction[
                (skeleton[1] - 1) * 3 : (skeleton[1] - 1) * 3 + 3
            ]

            cv2.line(
                img,
                (int(pos_x_1), int(pos_y_1)),
                (int(pos_x_2), int(pos_y_2)),
                color=color,
                thickness=tl,
            )

        for idx in range(len(prediction) // 3):
            x, y, score = prediction[idx * 3 : (idx + 1) * 3]
            cv2.circle(img, (int(x), int(y)), radius=3, color=kpt_color, thickness=-1)

    return img
