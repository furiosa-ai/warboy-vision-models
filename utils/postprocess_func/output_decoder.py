from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.postprocess_func.cbox_decode import yolov5_box_decode, yolov8_box_decode
from utils.postprocess_func.cpose_decode import yolov5_pose_decode, yolov8_pose_decode
from utils.postprocess_func.cseg_decode import yolov8_segmentation_decode
from utils.postprocess_func.nms import non_max_suppression
from utils.postprocess_func.tracking.bytetrack import ByteTrack


## Base YOLO Decoder
class YOLO_Decoder:
    """
    Base class for yolo output decoder, providing foundational attributes.

    Attributes:
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        tracker (ByteTrack | None) : tracker object when use traking algorithm.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        stride (list) : stride values for yolo.
    """

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = True,
    ):
        self.iou_thres = float(iou_thres)
        self.conf_thres = float(conf_thres)
        self.tracker = ByteTrack() if use_tracker else None
        self.anchors, self.stride = (
            get_anchors(anchors)
            if anchors[0] is not None
            else (None, np.array([(2 ** (i + 3)) for i in range(3)], dtype=np.float32))
        )


## YOLO Object Detection Decoder
class ObjDetDecoder(YOLO_Decoder):
    """
    A integrated version of the object detection decoder class for YOLO, adding nms operator and scaling operators.

    Attributes:
        model_name (str)  : a model name of yolo (e.g., yolov8s, yolov8m ...)
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        use_trakcer (bool): whether to use tracking algorithm.

    Methods:
        __call__(output, context, org_input_shape): Decode the result of YOLO Model (Object Detection)

    Usage:
        decoder = ObjDetDecoder(model_name, conf_thres, iou_thres, anchors)
        output = decoder(model_outputs, contexts, org_input_shape)
    """

    def __init__(
        self,
        model_name: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = True,
    ):
        super().__init__(conf_thres, iou_thres, anchors, use_tracker)
        self.box_decoder = (
            BoxDecoderYOLOv8(self.stride, self.conf_thres)
            if check_model(model_name)
            else BoxDecoderYOLOv5(self.stride, self.conf_thres, self.anchors)
        )

    def __call__(
        self,
        model_outputs: List[np.ndarray],
        contexts,
        org_input_shape: Tuple[int, int],
    ):
        boxes_dec = self.box_decoder(model_outputs)

        outputs = non_max_suppression(boxes_dec, self.iou_thres)
        predictions = []

        ratio, dwdh = contexts["ratio"], contexts["pad"]
        for _, prediction in enumerate(outputs):
            prediction[:, :4] = scale_coords(
                prediction[:, :4], ratio, dwdh, org_input_shape
            )  # Box Result
            if self.tracker is not None:
                prediction = self.tracker(prediction[:, :6])
            predictions.append(prediction)
        return predictions


## YOLO Pose Estimation Decoder
class PoseEstDecoder(YOLO_Decoder):
    """
    A integrated version of the pose estimation decoder class for YOLO, adding nms operator and scaling operators.

    Attributes:
        model_name (str)  : a model name of yolo (e.g., yolov8s, yolov8m ...)
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        use_trakcer (bool): whether to use tracking algorithm.

    Methods:
        __call__(output, context, org_input_shape): Decode the result of YOLO Model (Pose Estimation)

    Usage:
        decoder = PoseEstDecoder(model_name, conf_thres, iou_thres, anchors)
        output = decoder(model_outputs, contexts, org_input_shape)
    """

    def __init__(
        self,
        model_name: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = True,
    ):
        super().__init__(conf_thres, iou_thres, anchors, use_tracker)
        self.pose_decoder = (
            PoseDecoderYOLOv8(self.stride, self.conf_thres)
            if check_model(model_name)
            else PoseDecoderYOLOv5(self.stride, self.conf_thres, self.anchors)
        )

    def __call__(
        self,
        model_outputs: List[np.ndarray],
        contexts,
        org_input_shape: Tuple[int, int],
    ):
        poses_dec = self.pose_decoder(model_outputs)
        predictions = non_max_suppression(poses_dec, self.iou_thres)

        ratio, dwdh = contexts["ratio"], contexts["pad"]
        for _, prediction in enumerate(predictions):
            prediction[:, :4] = scale_coords(
                prediction[:, :4], ratio, dwdh, org_input_shape
            )  # Box Result
            prediction[:, 5:] = scale_coords(
                prediction[:, 5:], ratio, dwdh, org_input_shape, step=3
            )  # Pose Result

        return predictions


## YOLO Instance Segmentation Decoder
class InsSegDecoder(YOLO_Decoder):
    """
    A integrated version of the instance segmentation decoder class for YOLO, adding nms operator and scaling operators.

    Attributes:
        model_name (str)  : a model name of yolo (e.g., yolov8s, yolov8m ...)
        conf_thres (float): confidence score threshold for outputs.
        iou_thres (float): iou score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        use_trakcer (bool): whether to use tracking algorithm.

    Methods:
        __call__(output, context, org_input_shape): Decode the result of YOLO Model (Instance Segementation)

    Usage:
        decoder = InsSegDecoder(model_name, conf_thres, iou_thres, anchors)
        output = decoder(model_outputs, contexts, org_input_shape)
    """

    def __init__(
        self,
        model_name: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = True,
    ):
        super().__init__(conf_thres, iou_thres, anchors, use_tracker)
        self.box_decoder = BoxDecoderYOLOv8(self.stride, self.conf_thres)

    def __call__(
        self,
        model_outputs: Sequence[np.ndarray],
        contexts,
        org_input_shape: Tuple[int, int],
    ):
        proto = model_outputs[-1][0]
        model_outputs = model_outputs[:-1]
        ins_seg_dec = self.box_decoder(model_outputs, step=3)
        outputs = non_max_suppression(ins_seg_dec, self.iou_thres)

        predictions = []
        ratio, dwdh = contexts["ratio"], contexts["pad"]

        # scale proto
        h, w = proto.shape[1:]
        pad_w = dwdh[0] / 4
        pad_h = dwdh[1] / 4
        top, left = int(pad_h), int(pad_w)
        bottom, right = int(h - pad_h), int(w - pad_w)
        proto = proto[..., top:bottom, left:right]

        for _, prediction in enumerate(outputs):
            prediction[:, :4] = scale_coords(
                prediction[:, :4], ratio, dwdh, org_input_shape
            )  # Box Result
            ins_masks = process_mask(
                proto, prediction[:, 6:], prediction[:, :4], org_input_shape
            )

            bbox = prediction[:, :6]
            if self.tracker is not None:
                bbox = self.tracker(bbox)
            predictions.append((bbox, ins_masks))

        return predictions


##
class CDecoderBase:
    """
    Base class for decoder, providing foundational attributes.

    Attributes:
        conf_thres (float): confidence score threshold for outputs.
        anchors (list | none) : anchor values (anchor for yolov8 or yolov9 is None).
        stride (list) : stride values for yolo.
    """

    def __init__(self, stride: np.ndarray, conf_thres: float, anchors=None) -> None:
        self.stride = stride
        self.conf_thres = conf_thres
        self.anchors = anchors
        self.reg_max = 16


class BoxDecoderYOLOv8(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats: List[np.ndarray], step: int = 2):
        feats_box, feats_cls = feats[0::step], feats[1::step]
        feats_extra = None
        if step == 3:
            feats_extra = feats[2::step]

        out_boxes_batched = yolov8_box_decode(
            self.stride,
            self.conf_thres,
            self.reg_max,
            feats_box,
            feats_cls,
            feats_extra,
        )
        return out_boxes_batched


class BoxDecoderYOLOv5(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, feats: List[np.ndarray]):
        out_boxes_batched = yolov5_box_decode(
            self.anchors, self.stride, self.conf_thres, feats
        )
        return out_boxes_batched


class PoseDecoderYOLOv8(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_pose = 17

    def __call__(self, feats: List[np.ndarray]):
        feats_box, feats_cls, feats_pose = feats[0::3], feats[1::3], feats[2::3]
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


class PoseDecoderYOLOv5(CDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_pose = 17

    def __call__(self, feats: List[np.ndarray]):
        out_boxes_batched = yolov5_pose_decode(
            self.anchors, self.stride, self.conf_thres, self.num_pose, feats
        )
        return out_boxes_batched


## Useful functions for output decoder


def check_model(model_name: str) -> bool:
    if "yolov8" in model_name or "yolov9" in model_name:
        return True
    return False


def scale_coords(
    coords: List[np.ndarray],
    ratio: float,
    pad: Tuple[float, float],
    org_input_shape: Tuple[int, int],
    step: int = 2,
) -> np.ndarray:
    ## Scale the result values to fit original image
    coords[:, 0::step] = (1 / ratio) * (coords[:, 0::step] - pad[0])
    coords[:, 1::step] = (1 / ratio) * (coords[:, 1::step] - pad[1])

    ## Clip out-of-bounds values
    coords[:, 0::step] = np.clip(coords[:, 0::step], 0, org_input_shape[1])
    coords[:, 1::step] = np.clip(coords[:, 1::step], 0, org_input_shape[0])

    return coords


def process_mask(proto, mask_in, bbox, shape):
    c, mh, mw = proto.shape
    masks = yolov8_segmentation_decode(mask_in, proto)
    if len(masks) != 0:
        masks = F.interpolate(
            torch.from_numpy(masks[None]), shape, mode="bilinear", align_corners=False
        )[0]
        masks = masks.numpy()
        masks = _crop_mask(masks, bbox)
    return masks >= 0.5


def _crop_mask(masks, boxes):
    _, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], [1, 2, 3], axis=1)
    r = np.arange(w, dtype=np.float32)[None, None, :]
    c = np.arange(h, dtype=np.float32)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def get_anchors(anchors):
    num_layers = len(anchors)
    anchors = np.reshape(np.array(anchors, dtype=np.float32), (num_layers, -1, 2))
    stride = np.array([2 ** (i + 3) for i in range(num_layers)], dtype=np.float32)
    anchors /= np.reshape(stride, (-1, 1, 1))
    return anchors, stride
