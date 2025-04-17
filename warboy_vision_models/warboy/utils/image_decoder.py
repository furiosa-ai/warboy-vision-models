from typing import Callable, List

import cv2

from warboy_vision_models.warboy.utils.queue import (
    PipeLineQueue,
    QueueClosedError,
    StopSig,
)
from warboy_vision_models.warboy.yolo.preprocess import YoloPreProcessor


class ImageListDecoder:
    def __init__(
        self,
        image_list: List,
        stream_mux: PipeLineQueue,
        frame_mux: PipeLineQueue,
        preprocess_function: Callable = YoloPreProcessor(),
    ):
        self.image_paths = PipeLineQueue()
        for image in image_list:
            self.image_paths.put(image.image_info)
        self.image_paths.put(StopSig)

        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux

    def run(self):
        img_idx = 0
        while True:
            try:
                image_path = self.image_paths.get()
                img = cv2.imread(image_path)
                input_, context = self.preprocessor(img)
                self.stream_mux.put((input_, img_idx))
                self.frame_mux.put((img, context, img_idx))
                img_idx += 1

            except QueueClosedError:
                break

        self.stream_mux.put(StopSig)
        self.frame_mux.put(StopSig)
        return
