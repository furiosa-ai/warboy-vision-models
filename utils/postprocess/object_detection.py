from typing import List, Sequence, Tuple

import cv2
import numpy as np

from utils.postprocess.settings import COLORS
from utils.postprocess.yolo_object_detection.postprocess import (
    YOLOv5PostProcessor,
    YOLOv8PostProcessor,
)


class ObjDetPostProcess:
    def __init__(self, model_name, model_cfg):
        if "yolov8" in model_name:
            self.postprocess = YOLOv8PostProcessor(**model_cfg)
        elif "yolov7" in model_name or "yolov5" in model_name:
            self.postprocess = YOLOv5PostProcessor(**model_cfg)
        else:
            raise "Unsupported Model (currently, only YOLO models are supported)"
        self.class_names = self.postprocess.class_names

    def __call__(
        self,
        outputs: np.ndarray,
        preproc_params: Tuple[float, Tuple[float, float]],
        img: np.ndarray,
        **args,
    ) -> np.ndarray:
        predictions = self.postprocess(outputs, preproc_params)
        assert len(predictions) == 1, f"{len(predictions)}!=1"
        predictions = predictions[0]
        num_predictions = predictions.shape[0]

        if num_predictions == 0:
            return img

        bboxed_img = self.draw_bbox(img, predictions)
        return bboxed_img

    def draw_bbox(self, img: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        for i, box in enumerate(predictions):
            x0, y0, x1, y1 = [int(i) for i in box[:4]]
            mbox = np.array([x0, y0, x1, y1])
            mbox = mbox.round().astype(np.int32).tolist()
            score = box[4]
            class_id = int(box[5])

            color = COLORS[class_id % len(COLORS)]
            label = f"{self.class_names[class_id]} ({score:.2f})"
            img = self.plot_one_box(mbox, img, color, label)
        return img

    def plot_one_box(self, box, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
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
