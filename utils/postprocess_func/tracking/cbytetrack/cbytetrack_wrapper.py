import ctypes
import os
import platform

import numpy as np
import torch

if platform.uname()[0] == "Windows":
    _lib_ext = "dll"
elif platform.uname()[0] == "Linux":
    _lib_ext = "so"
else:
    _lib_ext = "dylib"

_clib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "build", f"libbytetrack.{_lib_ext}")
)


def _init():
    vp = ctypes.c_void_p
    u32 = ctypes.c_uint32
    f32 = ctypes.c_float

    f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

    _clib.ByteTrackNew.argtypes = [f32, f32, f32, u32, u32]
    _clib.ByteTrackNew.restype = vp

    _clib.ByteTrackDelete.argtypes = [vp]
    _clib.ByteTrackDelete.restype = None

    _clib.ByteTrackUpdate.argtypes = [vp, f32p, u32, f32p, u32]
    _clib.ByteTrackUpdate.restype = u32

    _clib.get_tracked_tracks_count.argtypes = [vp]
    _clib.get_tracked_tracks_count.restype = u32

    _clib.get_lost_tracks_count.argtypes = [vp]
    _clib.get_lost_tracks_count.restype = u32

    _clib.get_removed_tracks_count.argtypes = [vp]
    _clib.get_removed_tracks_count.restype = u32

    _clib.clear_buffer.argtypes = [vp]
    _clib.clear_buffer.restype = None

    _clib.clear_lost.argtypes = [vp]
    _clib.clear_lost.restype = None


class CByteTrack:
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        high_thresh=None,
        framerate=30,
    ) -> None:
        if high_thresh is None:
            high_thresh = track_thresh
        self.obj = _clib.ByteTrackNew(
            track_thresh, high_thresh, match_thresh, framerate, track_buffer
        )
        self.buffer = None

    @property
    def tracked_tracks_count(self):
        return _clib.get_tracked_tracks_count(self.obj)

    @property
    def lost_tracks_count(self):
        return _clib.get_lost_tracks_count(self.obj)

    @property
    def removed_tracks_count(self):
        return _clib.get_removed_tracks_count(self.obj)

    def clear_buffer(self):
        return _clib.clear_buffer(self.obj)

    def clear_lost(self):
        return _clib.clear_lost(self.obj)

    def update(self, boxes):
        assert boxes.shape[1] >= 5, "Need xyxyp format"

        num_boxes = boxes.shape[0]
        n_extra = boxes.shape[1] - 5

        if self.buffer is None or self.buffer.shape[1] != boxes.shape[1] + 1:
            self.buffer = np.zeros((int(1e5), boxes.shape[1] + 1), dtype=np.float32)

        num_tracks = _clib.ByteTrackUpdate(
            self.obj, np.ascontiguousarray(boxes), num_boxes, self.buffer, n_extra
        )
        boxes_track = self.buffer[:num_tracks].copy()
        return boxes_track

    def close(self):
        if self.obj is not None:
            _clib.ByteTrackDelete(self.obj)
            self.obj = None

    def __del__(self):
        self.close()


_init()
