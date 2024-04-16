import multiprocessing as mp
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils.mp_queue import MpQueue, QueueClosedError


class JobHandler:
    def __init__(self, handlers):
        self.handlers = handlers

    def start(self):
        for proc in self.handlers:
            proc.start()

    def join(self):
        for proc in self.handlers:
            proc.join()


class InputHandler(JobHandler):
    """ """

    def __init__(
        self,
        input_video_paths: List[str],
        output_dir: str,
        input_queue: MpQueue,
        preprocessor,
        input_shape: Tuple[int, int] = (640, 640),
    ):
        self.video_handlers = [
            mp.Process(
                target=self.video_to_input,
                args=(input_video_path, video_idx, input_queue),
            )
            for video_idx, input_video_path in enumerate(input_video_paths)
        ]
        super().__init__(self.video_handlers)

        self.output_dir = output_dir  # Output Path for saving Result
        self.input_shape = input_shape  # Shape for Model
        self.preprocessor = preprocessor  # Preprocessor for Input of Model

    def video_to_input(
        self, input_video_path: str, video_idx: int, input_queue: MpQueue
    ) -> None:
        video_name = get_video_name(input_video_path)
        output_path = os.path.join(self.output_dir, "_input", video_name)

        os.makedirs(output_path)

        cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
        img_idx = 0

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                make_ending_signal(output_path, img_idx)
                break

            input_, contexts = self.preprocessor(frame, self.input_shape)
            try:
                input_queue.put((input_, contexts, img_idx, video_idx))
            except queue.Full:
                time.sleep(0.001)
                input_queue.put((input_, contexts, img_idx, video_idx))

            img_path = os.path.join(output_path, "%010d.bmp" % img_idx)
            cv2.imwrite(img_path, frame)

            img_idx += 1

        if cap.isOpened():
            cap.release()
        return


class OutputHandler(JobHandler):
    """ """

    def __init__(
        self,
        input_video_paths: List[str],
        output_dir: str,
        output_queues: List[MpQueue],
        postprocessor,
        draw_fps: bool = True,
    ):
        self.output_handlers = [
            mp.Process(
                target=self.output_to_img,
                args=(input_video_path, output_queues[video_idx]),
            )
            for video_idx, input_video_path in enumerate(input_video_paths)
        ]
        super().__init__(self.output_handlers)
        self.output_dir = output_dir
        self.postprocessor = postprocessor
        self.draw_fps = draw_fps

    def output_to_img(self, input_video_path: str, output_queue: MpQueue):
        video_name = get_video_name(input_video_path)
        input_img_dir = os.path.join(self.output_dir, "_input", video_name)
        output_path = os.path.join(self.output_dir, "_output", video_name)

        os.makedirs(output_path)
        max_ = 0

        # Information for FPS
        start_time = time.time()
        FPS = 0.0
        completed = 0
        last = 0

        while True:
            try:
                predictions, contexts, img_idx = output_queue.get()
            except QueueClosedError:
                break

            input_img = cv2.imread(os.path.join(input_img_dir, "%010d.bmp" % img_idx))
            max_ = max(max_, img_idx + 1)
            if os.path.exists(os.path.join(input_img_dir, "%010d.csv" % max_)):
                make_ending_signal(output_path, max_)

            ## Make output image from predictions
            output_img = self.postprocessor(predictions, contexts, input_img)

            if self.draw_fps:
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.3:
                    FPS = (completed - last) / elapsed_time
                    start_time = time.time()
                    last = completed
                output_img = put_fps_to_img(output_img, FPS)

            dummy_path = os.path.join(output_path, ".%010d.bmp" % img_idx)
            img_path = os.path.join(output_path, "%010d.bmp" % img_idx)
            cv2.imwrite(dummy_path, output_img)
            os.rename(dummy_path, img_path)
            completed += 1

        if os.path.exists(input_img_dir):
            subprocess.run(["rm", "-rf", input_img_dir])

        return


def get_video_name(video_path: str) -> str:
    return (video_path.split("/")[-1]).split(".")[0]


def put_fps_to_img(output_img: np.ndarray, FPS: float):
    h, w, _ = output_img.shape
    org = (int(0.05 * h), int(0.05 * w))
    scale = int(org[1] * 0.07)

    output_img = cv2.putText(
        output_img,
        f"FPS: {FPS:.2f}",
        (int(0.05 * h), int(0.05 * w)),
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 0, 0),
        scale,
        cv2.LINE_AA,
    )
    return output_img


def make_ending_signal(output_path, idx):
    end_file = Path(os.path.join(output_path, "%010d.csv" % idx))
    end_file.touch(exist_ok=True)
    return
