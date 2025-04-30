import re
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from .cbox_decode import yolov5_box_decode, yolov8_box_decode
from .cpose_decode import yolov5_pose_decode, yolov8_pose_decode
from .tracking.bytetrack import ByteTrack

# from furiosa.yolo.tracking.bytetrack import ByteTrack


def non_max_suppression(
    prediction: List[np.ndarray], iou_thres: float = 0.45, class_agnostic=True
) -> List[np.ndarray]:
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    output = []
    for x in prediction:
        t = torch.from_numpy(x[:, :6])
        i = nms(t[:, :4], t[:, 4], iou_thres)
        if len(i) == 1:
            output.append(np.array([x[i]]))
        else:
            output.append(x[i])
    return output


class anchor_decoder:
    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = False,
    ):
        self.iou_thres = float(iou_thres)
        self.conf_thres = float(conf_thres)
        self.tracker = ByteTrack() if use_tracker else None
        self.anchors, self.stride = (
            self.get_anchors(anchors)
            if anchors[0] is not None
            else (None, np.array([(2 ** (i + 3)) for i in range(3)], dtype=np.float32))
        )

    def get_anchors(self, anchors):
        num_layers = len(anchors)
        anchors = np.reshape(np.array(anchors, dtype=np.float32), (num_layers, -1, 2))
        stride = np.array([2 ** (i + 3) for i in range(num_layers)], dtype=np.float32)
        anchors /= np.reshape(stride, (-1, 1, 1))
        return anchors, stride


## YOLO Object Detection Decoder
class object_detection_anchor_decoder(anchor_decoder):
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
        if re.search(r"yolov5.*6u", model_name):
            self.box_decoder.stride = np.array(
                [(2 ** (i + 3)) for i in range(4)], dtype=np.float32
            )
        print(conf_thres, iou_thres, anchors, use_tracker)

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
            try:
                prediction[:, :4] = scale_coords(
                    prediction[:, :4], ratio, dwdh, org_input_shape
                )  # Box Result
                if self.tracker is not None:
                    prediction = self.tracker(prediction[:, :6])
                predictions.append(prediction)
            except Exception as e:
                continue
        return predictions


## YOLO Pose Estimation Decoder
class pose_estimation_anchor_decoder(anchor_decoder):
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
        decoder = pose_estimation_anchor_decoder(model_name, conf_thres, iou_thres, anchors)
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
        print(self.conf_thres, self.iou_thres, use_tracker)

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
class instance_segment_anchor_decoder(anchor_decoder):
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
        decoder = instance_segment_anchor_decoder(model_name, conf_thres, iou_thres, anchors)
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
                torch.from_numpy(proto),
                torch.from_numpy(prediction[:, 6:]),
                prediction[:, :4],
                org_input_shape,
            )
            # ins_masks = process_mask(
            #    proto, prediction[:, 6:], prediction[:, :4], org_input_shape
            # )

            bbox = prediction[:, :6]
            # if self.tracker is not None:
            #    bbox = self.tracker(bbox)
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
    import re

    if (
        "yolov8" in model_name
        or "yolov9" in model_name
        or re.search(r"yolov5.*u", model_name)
    ):
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
    # masks = yolov8_segmentation_decode(mask_in, proto)
    masks = (mask_in @ proto.contiguous().view(c, -1)).sigmoid().view(-1, mh, mw)
    if len(masks) != 0:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[
            0
        ]
        masks = masks.numpy()
        masks = _crop_mask(masks, bbox)
    return masks >= 0.5


def _crop_mask(masks, boxes):
    _, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], [1, 2, 3], axis=1)
    r = np.arange(w, dtype=np.float32)[None, None, :]
    c = np.arange(h, dtype=np.float32)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
