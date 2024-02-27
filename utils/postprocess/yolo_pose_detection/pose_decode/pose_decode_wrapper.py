import ctypes
import os

import numpy as np

_clib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cpose_decode.so"))


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
        const float* const feat_box, const float* const feat_cls, const float* const feat_pose, const uint32_t batch_size,
        const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max, const uint32_t npose,
        float* const out_batch, uint32_t* out_batch_pos,
    """

    _clib.yolov8_pose_decode_feat.argtypes = [
        f32,  # stride
        f32,  # conf_thres
        u32,  # max_boxes
        f32p,  # feat_box
        f32p,  # feat_cls
        f32p,  # feat_pose
        u32,  # batch_size
        u32,  # ny
        u32,  # nx
        u32,  # nc
        u32,  # reg_max
        u32,  # npose
        f32p,  # out_batch
        u32p,  # out_batch_pos
    ]
    _clib.yolov8_pose_decode_feat.restype = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def _yolov8_pose_decode_feat(
    stride,
    conf_thres,
    reg_max,
    num_pose,
    max_boxes,
    feat_box,
    feat_cls,
    feat_pose,
    out_batch,
    out_batch_pos,
):
    feat_box = feat_box.transpose(0, 2, 3, 1)
    feat_cls = feat_cls.transpose(0, 2, 3, 1)
    feat_pose = feat_pose.transpose(0, 2, 3, 1)
    feat_pose[..., 2::3] = sigmoid(feat_pose[..., 2::3])

    assert feat_box.shape[:3] == feat_cls.shape[:3] and feat_pose.shape[:3] == feat_cls.shape[:3]
    bs, ny, nx, num_box_params = feat_box.shape
    bs, ny, nx, num_pose_params = feat_pose.shape

    assert num_box_params == 4 * reg_max and num_pose_params == 3 * num_pose
    _, _, _, nc = feat_cls.shape

    if isinstance(feat_box, np.ndarray):
        _clib.yolov8_pose_decode_feat(
            stride,
            conf_thres,
            max_boxes,
            np.ascontiguousarray(feat_box),
            np.ascontiguousarray(feat_cls),
            np.ascontiguousarray(feat_pose),
            bs,
            ny,
            nx,
            nc,
            reg_max,
            num_pose,
            out_batch,
            out_batch_pos,
        )
    else:
        raise Exception(type(feat_box))


def yolov8_pose_decode(stride, conf_thres, reg_max, num_pose, feats_box, feats_cls, feats_pose):
    bs = feats_box[0].shape[0]
    max_boxes = int(1e4)

    out_batch = np.empty((bs, max_boxes, (5 + num_pose * 3)), dtype=np.float32)
    out_batch_pos = np.zeros(bs, dtype=np.uint32)

    for l, (feat_box, feat_cls, feat_pose) in enumerate(zip(feats_box, feats_cls, feats_pose)):
        _yolov8_pose_decode_feat(
            stride[l],
            conf_thres,
            reg_max,
            num_pose,
            max_boxes,
            feat_box,
            sigmoid(feat_cls),
            feat_pose,
            out_batch,
            out_batch_pos,
        )

    out_boxes_batched = [
        boxes[: (pos // (5 + num_pose * 17))] for boxes, pos in zip(out_batch, out_batch_pos)
    ]

    return out_boxes_batched


_init()
