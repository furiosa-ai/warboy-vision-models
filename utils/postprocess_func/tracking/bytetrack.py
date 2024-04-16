import sys

import numpy as np

from utils.postprocess_func.tracking.cbytetrack import CByteTrack


class ByteTrack:
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        clear_track_interval=30,
    ):
        self.tracker = CByteTrack(track_thresh, track_buffer, match_thresh)
        self.max_track_count = None
        self.clear_track_interval = clear_track_interval
        self.frame_id = 0

    def clear_buffer(self):
        self.tracker.clear_buffer()

    def __call__(self, boxes):
        if (self.frame_id % self.clear_track_interval) == 0:
            self.tracker.clear_buffer()

        pred = self.tracker.update(boxes)

        if len(pred) == 0:
            pred = np.zeros((0, boxes.shape[1] + 1))

        self.frame_id += 1
        return pred
