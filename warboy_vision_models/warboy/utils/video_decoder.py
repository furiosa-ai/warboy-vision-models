from typing import Callable

import cv2

from warboy_vision_models.warboy.utils.queue import PipeLineQueue, StopSig
from warboy_vision_models.warboy.yolo.preprocess import YoloPreProcessor


class VideoDecoder:
    def __init__(
        self,
        video_path: str,
        stream_mux: PipeLineQueue,
        frame_mux: PipeLineQueue,
        preprocess_function: Callable = YoloPreProcessor(),
        recursive: bool = False,
    ):
        self.video_path = video_path
        self.reader = None
        self.recursive = recursive
        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux

    def run(self):
        img_idx = 0
        self.reader = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        while True:
            try:
                hasFrame, frame = self.reader.read()
                if not hasFrame:
                    if self.recursive:
                        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.reader.release()
                        break

                input_, context = self.preprocessor(frame)
                self.stream_mux.put((input_, img_idx))
                self.frame_mux.put((frame, context, img_idx))
                img_idx += 1

            except Exception as e:
                print(e, self.video_path)
                break

        self.stream_mux.put(StopSig)
        self.frame_mux.put(StopSig)
        print(f"End Video!! -> {self.video_path}")
        return
