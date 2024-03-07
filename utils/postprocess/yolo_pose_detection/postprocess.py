from typing import List, Sequence, Tuple

import numpy as np
import torch

from utils.postprocess.yolo_pose_detection.nms import non_max_suppression
from utils.postprocess.yolo_pose_detection.pose_decode import yolov8_pose_decode, yolov5_pose_decode


class PoseDecoderBase:
    def __init__(self, stride, conf_thres, anchors=None) -> None:
        self.stride = stride
        self.conf_thres = conf_thres
        self.reg_max = 16
        self.num_pose = 17
        self.anchors = anchors


class PoseDecoderYOLOv8(PoseDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats_box, feats_cls, feats_pose):
        out_boxes_batched = yolov8_pose_decode(
            self.stride,
            self.conf_thres,
            self.reg_max,
            self.num_pose,
            feats_box,
            feats_cls,
            feats_pose,
        )
        return out_boxes_batched


class YOLOv8PostProcessor:
    def __init__(
        self,
        conf_thres: float = 0.65,
        iou_thres: float = 0.45,
        input_shape: Tuple[int, int] = (640, 640),
        class_names: List[str] = None,
        anchors=None,
    ):
        self.stride = torch.FloatTensor([(2 ** (i + 3)) for i in range(3)])
        self.class_names = class_names
        self.nc = len(class_names)
        self.classes = [i for i in range(1, self.nc + 1)]
        self.reg_max = 16
        self.input_shape = input_shape
        self.no = self.nc + self.reg_max * 4
        self.iou_thres = float(iou_thres)
        self.pose_decoder = PoseDecoderYOLOv8(stride=self.stride, conf_thres=float(conf_thres))

    def __call__(self, model_outputs: Sequence[np.ndarray], contexts):
        box_info = model_outputs[:6]
        boxes_dec = self.pose_decoder(box_info[0::2], box_info[1::2], model_outputs[6:])
        outputs = non_max_suppression(boxes_dec, self.iou_thres)

        for _, prediction in enumerate(outputs):
            ratio, dwdh = contexts["ratio"], contexts["pad"]
            prediction[:, [0, 2]] = (1 / ratio) * (prediction[:, [0, 2]] - dwdh[0])
            prediction[:, [1, 3]] = (1 / ratio) * (prediction[:, [1, 3]] - dwdh[1])
            prediction[:, 5 + 0 :: 3] = (1 / ratio) * (prediction[:, 5 + 0 :: 3] - dwdh[0])
            prediction[:, 5 + 1 :: 3] = (1 / ratio) * (prediction[:, 5 + 1 :: 3] - dwdh[1])

        return outputs


class PoseDecoderYOLOv5(PoseDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats):
        out_boxes_batched = yolov5_pose_decode(
            self.anchors,
            self.stride,
            self.conf_thres,
            self.num_pose,
            feats
        )
        return out_boxes_batched


class YOLOv5PostProcessor:
    def __init__(
        self,
        conf_thres: float = 0.65,
        iou_thres: float = 0.45,
        input_shape: Tuple[int, int] = (640, 640),
        class_names: List[str] = None,
        anchors=None,
    ):
        self.class_names = class_names
        self.nc = len(class_names)
        self.classes = [i for i in range(1, self.nc + 1)]
        self.iou_thres = iou_thres
        self.input_shape = input_shape
        self.num_layers = len(anchors)
        self.anchors, self.stride = self.get_anchors(anchors)
        self.anchor_per_layer_count = self.anchors.shape[1]
        self.pose_decoder = PoseDecoderYOLOv5(
            stride=self.stride, conf_thres=conf_thres, anchors=self.anchors
        )

    def __call__(self, model_outputs: Sequence[np.ndarray], contexts):
        poses_dec = self.pose_decoder(model_outputs)
        outputs = non_max_suppression(poses_dec, self.iou_thres)
        for _, prediction in enumerate(outputs):
            ratio, dwdh = contexts["ratio"], contexts["pad"]
            prediction[:, [0, 2]] = (1 / ratio) * (prediction[:, [0, 2]] - dwdh[0])
            prediction[:, [1, 3]] = (1 / ratio) * (prediction[:, [1, 3]] - dwdh[1])
            prediction[:, 6 + 0 :: 3] = (1 / ratio) * (prediction[:, 6 + 0 :: 3] - dwdh[0])
            prediction[:, 6 + 1 :: 3] = (1 / ratio) * (prediction[:, 6 + 1 :: 3] - dwdh[1])

        return outputs


    def get_anchors(self, anchors):
        num_layers = len(anchors)
        anchors = np.reshape(np.array(anchors, dtype=np.float32), (num_layers, -1, 2))
        stride = np.array([2 ** (i + 3) for i in range(num_layers)], dtype=np.float32)
        anchors /= np.reshape(stride, (-1, 1, 1))
        return anchors, stride
