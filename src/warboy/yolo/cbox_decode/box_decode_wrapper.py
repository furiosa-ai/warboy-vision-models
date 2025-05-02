import ctypes
import os
from typing import List, Union

import numpy as np

_clib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cbox_decode.so"))


def _init():
    u8 = ctypes.c_uint8
    i32 = ctypes.c_int32
    u32 = ctypes.c_uint32
    f32 = ctypes.c_float

    u8p = np.ctypeslib.ndpointer(dtype=u8, flags="C_CONTIGUOUS")
    u32p = np.ctypeslib.ndpointer(dtype=u32, flags="C_CONTIGUOUS")
    f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

    """
        const float stride, const float conf_thres, const uint32_t max_boxes,
        const uint8_t* const feat_box, const uint8_t* const feat_cls, const uint32_t batch_size,
        const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max,
        float* const out_batch, uint32_t* out_batch_pos,
        const float quant_scale_box, const int32_t quant_zero_point_box,
        const float quant_scale_cls, const int32_t quant_zero_point_cls
    """

    _clib.yolov8_box_decode_feat.argtypes = [
        f32,  # stride
        f32,  # conf_thres
        u32,  # max_boxes
        f32p,  # feat_box
        f32p,  # feat_cls
        f32p,  # feat_extra
        u32,  # batch_size
        u32,  # ny
        u32,  # nx
        u32,  # nc
        u32,  # reg_max
        u32,  # n_extra
        f32p,  # out_batch
        u32p,  # out_batch_pos
    ]
    _clib.yolov8_box_decode_feat.restype = None

    _clib.yolov5_box_decode_feat.argtypes = [
        f32p,  # anchors
        u32,  # num_anchors
        f32,  # stride
        f32,  # conf_thres
        u32,  # max_boxes
        f32p,  # feat
        u32,  # batch_size
        u32,  # ny
        u32,  # nx
        u32,  # no
        f32p,  # out_batch
        u32p,  # out_batch_pos
    ]
    _clib.yolov5_box_decode_feat.restype = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def _yolov8_box_decode_feat(
    stride,
    conf_thres,
    reg_max,
    max_boxes,
    feat_box,
    feat_cls,
    feat_extra,
    out_batch,
    out_batch_pos,
):
    feat_box = feat_box.transpose(0, 2, 3, 1)
    feat_cls = feat_cls.transpose(0, 2, 3, 1)
    assert feat_box.shape[:3] == feat_cls.shape[:3]

    bs, ny, nx, num_box_params = feat_box.shape
    assert num_box_params == 4 * reg_max

    _, _, _, nc = feat_cls.shape

    if feat_extra is None:
        feat_extra = np.zeros([0], dtype=np.float32)
        n_extra = 0
    else:
        feat_extra = feat_extra.transpose(0, 2, 3, 1)
        _, _, _, n_extra = feat_extra.shape

    if isinstance(feat_box, np.ndarray):
        _clib.yolov8_box_decode_feat(
            stride,
            conf_thres,
            max_boxes,
            np.ascontiguousarray(feat_box),
            np.ascontiguousarray(feat_cls),
            np.ascontiguousarray(feat_extra),
            bs,
            ny,
            nx,
            nc,
            reg_max,
            n_extra,
            out_batch,
            out_batch_pos,
        )
    else:
        raise Exception(type(feat_box))


def yolov8_box_decode(
    stride: np.ndarray,
    conf_thres: float,
    reg_max: int,
    feats_box: np.ndarray,
    feats_cls: np.ndarray,
    feats_extra: Union[np.ndarray, None] = None,
) -> List[np.ndarray]:
    bs = feats_box[0].shape[0]
    max_boxes = int(1e4)
    n_extra = feats_extra[0].shape[1] if feats_extra is not None else 0
    out_batch = np.empty((bs, max_boxes, (6 + n_extra)), dtype=np.float32)
    out_batch_pos = np.zeros(bs, dtype=np.uint32)

    for l, (feat_box, feat_cls) in enumerate(zip(feats_box, feats_cls)):
        _yolov8_box_decode_feat(
            stride[l],
            conf_thres,
            reg_max,
            max_boxes,
            feat_box,
            sigmoid(feat_cls),
            feats_extra[l] if feats_extra is not None else None,
            out_batch,
            out_batch_pos,
        )
    out_boxes_batched = [
        boxes[: (pos // (6 + n_extra))] for boxes, pos in zip(out_batch, out_batch_pos)
    ]
    return out_boxes_batched


def _yolov5_box_decode_feat(
    anchors, stride, conf_thres, max_boxes, feat, out_batch, out_batch_pos
):
    bs, na, ny, nx, no = feat.shape
    if isinstance(feat, np.ndarray):
        _clib.yolov5_box_decode_feat(
            anchors.reshape(-1),
            na,
            stride,
            conf_thres,
            max_boxes,
            feat.reshape(-1),
            bs,
            ny,
            nx,
            no,
            out_batch,
            out_batch_pos,
        )
    else:
        raise Exception(type(feat))


def yolov5_box_decode(
    anchors: np.ndarray, stride: np.ndarray, conf_thres: float, feats: np.ndarray
) -> List[np.ndarray]:
    bs = feats[0].shape[0]
    max_boxes = int(1e4)
    out_batch = np.empty((bs, max_boxes, 6), dtype=np.float32)
    out_batch_pos = np.zeros(bs, dtype=np.uint32)
    for l, feat in enumerate(feats):
        bs, _, ny, nx = feat.shape
        feat = feat.reshape(bs, 3, -1, ny, nx).transpose(0, 1, 3, 4, 2)
        _yolov5_box_decode_feat(
            anchors[l],
            stride[l],
            conf_thres,
            max_boxes,
            sigmoid(feat),
            out_batch,
            out_batch_pos,
        )

    out_boxes_batched = [
        boxes[: (pos // 6)] for boxes, pos in zip(out_batch, out_batch_pos)
    ]

    return out_boxes_batched


_init()
