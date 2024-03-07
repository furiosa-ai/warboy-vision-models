from typing import List, Sequence, Tuple

import numpy as np
import torch

from utils.postprocess.yolo_object_detection.box_decode import yolov5_box_decode, yolov8_box_decode
from utils.postprocess.yolo_object_detection.nms import non_max_suppression


class BoxDecoderBase:
    def __init__(self, stride, conf_thres, anchors=None) -> None:
        self.stride = stride
        self.anchors = anchors
        self.conf_thres = conf_thres
        self.reg_max = 16


class BoxDecoderYOLOv8(BoxDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats_box, feats_cls, feats_extra=None):
        out_boxes_batched = yolov8_box_decode(
            self.stride, self.conf_thres, self.reg_max, feats_box, feats_cls, feats_extra
        )
        return out_boxes_batched


class BoxDecoderYOLOv5(BoxDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats):
        out_boxes_batched = yolov5_box_decode(self.anchors, self.stride, self.conf_thres, feats)
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
        self.box_decoder = BoxDecoderYOLOv8(stride=self.stride, conf_thres=float(conf_thres))

    def __call__(self, model_outputs: Sequence[np.ndarray], contexts):
        boxes_dec = self.box_decoder(model_outputs[0::2], model_outputs[1::2])
        boxes = non_max_suppression(boxes_dec, self.iou_thres)

        for _, prediction in enumerate(boxes):
            ratio, dwdh = contexts["ratio"], contexts["pad"]
            prediction[:, [0, 2]] = (1 / ratio) * (prediction[:, [0, 2]] - dwdh[0])
            prediction[:, [1, 3]] = (1 / ratio) * (prediction[:, [1, 3]] - dwdh[1])

        return boxes


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
        self.input_shape = input_shape
        self.num_layers = len(anchors)
        self.anchors, self.stride = self.get_anchors(anchors)
        self.anchor_per_layer_count = self.anchors.shape[1]
        self.box_decoder = BoxDecoderYOLOv5(
            stride=self.stride, conf_thres=conf_thres, anchors=self.anchors
        )

    def __call__(self, model_outputs: Sequence[np.ndarray], contexts):
        boxes_dec = self.box_decoder(model_outputs)
        boxes = non_max_suppression(boxes_dec, self.iou_thres)

        for _, prediction in enumerate(boxes):
            ratio, dwdh = contexts["ratio"], contexts["pad"]
            prediction[:, [0, 2]] = (1 / ratio) * (prediction[:, [0, 2]] - dwdh[0])
            prediction[:, [1, 3]] = (1 / ratio) * (prediction[:, [1, 3]] - dwdh[1])

        return boxes

    def get_anchors(self, anchors):
        num_layers = len(anchors)
        anchors = np.reshape(np.array(anchors, dtype=np.float32), (num_layers, -1, 2))
        stride = np.array([2 ** (i + 3) for i in range(num_layers)], dtype=np.float32)
        anchors /= np.reshape(stride, (-1, 1, 1))
        return anchors, stride
