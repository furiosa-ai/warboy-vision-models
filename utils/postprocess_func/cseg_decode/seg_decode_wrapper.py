import ctypes
import os

import numpy as np

_clib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cseg_decode.so"))


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

    _clib.yolov8_seg_decode.argtypes = [
        f32p,  # mask_in
        f32p,  # proto
        u32,  # c
        u32,  # mh
        u32,  # mw
        u32,  # num_out
        f32p,  # output
        u32p,  # output_pos
    ]
    _clib.yolov8_seg_decode.restype = None


def yolov8_segmentation_decode(mask_in, proto):
    c, mh, mw = proto.shape
    proto = proto.reshape(c, -1)

    num_out, _ = mask_in.shape
    output = np.empty((num_out, mh, mw), dtype=np.float32)
    out_pos = np.zeros(num_out, dtype=np.uint32)
    if isinstance(mask_in, np.ndarray):
        _clib.yolov8_seg_decode(
            np.ascontiguousarray(mask_in),
            np.ascontiguousarray(proto),
            c,
            mh,
            mw,
            num_out,
            output,
            out_pos,
        )

    out_result = np.array(
        [boxes[:pos].reshape(mh, mw) for boxes, pos in zip(output, out_pos)]
    )
    return out_result


_init()
