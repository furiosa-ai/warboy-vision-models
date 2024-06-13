import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

from utils.mp_queue import MpQueue, QueueClosedError, QueueStopEle


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
    def __init__(
        self,
        input_videos_info: List[str],
        input_queues: List[Tuple[MpQueue, MpQueue]],
        frame_queues: List[MpQueue],
        preprocessor,
        input_shape: Tuple[int, int] = (640, 640),
    ):
        self.video_handlers = [
            mp.Process(
                target=self.video_to_input,
                args=(
                    input_video_info["input_path"],
                    video_idx,
                    input_queues[video_idx],
                    frame_queues[video_idx],
                    input_video_info["type"],
                    input_video_info["recursive"]
                ),
            )
            for video_idx, input_video_info in enumerate(input_videos_info)
        ]
        super().__init__(self.video_handlers)
        self.input_shape = input_shape  # Shape for Model
        self.preprocessor = preprocessor  # Preprocessor for Input of Model

    def video_to_input(
        self,
        input_video_path: str,
        video_idx: int,
        input_queue: Tuple[MpQueue, MpQueue],
        frame_queue: MpQueue,
        video_type: str = "file",
        recursive: bool = True,
    ) -> None:
        video_name = get_video_name(input_video_path)
        img_idx = 0
        while True:
            if video_type == "file":
                if not os.path.exists(input_video_path):
                    raise Exception(f"{input_video_path} Video File is not found !!")
                cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)  # video file
            elif video_type == "webcam":
                cap = cv2.VideoCapture(int(input_video_path))  # webcam
            else:
                raise Exception(f"This {video_type} type is currently not supported !!")
            
            while True:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    break

                frame_queue.put(frame)

                for input_shape, iq in zip(self.input_shape, input_queue):
                    input_, contexts = self.preprocessor(frame, input_shape)
                    iq.put((input_, contexts, img_idx, video_idx))
                img_idx += 1
            
            if cap.isOpened():
                cap.release()

            if not recursive:
                frame_queue.put(QueueStopEle)
                for iq in input_queue:
                    iq.put(QueueStopEle)
                break

        print(f"{video_idx}th-Video Process End..")
        return


class OutputHandler(JobHandler):
    def __init__(
        self,
        input_videos_info: List[Dict[str, Any]],
        output_queues: List[Tuple[MpQueue, MpQueue]],
        frame_queues: List[MpQueue],
        result_queues: List[MpQueue],
        postprocessor,
        draw_fps: bool = True,
    ):
        self.output_handlers = [
            mp.Process(
                target=self.output_to_img,
                args=(
                    input_video_info["input_path"],
                    output_queues[video_idx],
                    frame_queues[video_idx],
                    result_queues[video_idx],
                ),
            )
            for video_idx, input_video_info in enumerate(input_videos_info)
        ]
        super().__init__(self.output_handlers)
        self.postprocessor = postprocessor
        self.num_handler_process = 1
        self.draw_fps = draw_fps

    def output_to_img(
        self,
        input_video_path: str,
        output_queue: MpQueue,
        frame_queue: MpQueue,
        result_queue: MpQueue,
    ):
        current_idx = mp.Value("i", 0)
        output_handler_processes = [
            mp.Process(
                target=self._output_to_img,
                args=(
                    input_video_path,
                    output_queue,
                    frame_queue,
                    result_queue,
                    idx,
                    current_idx,
                ),
            )
            for idx in range(self.num_handler_process)
        ]
        for proc in output_handler_processes:
            proc.start()

        for proc in output_handler_processes:
            proc.join()

    def _output_to_img(
        self,
        input_video_path: str,
        output_queue: MpQueue,
        frame_queue: MpQueue,
        result_queue: MpQueue,
        current_idx,
        cidx,
    ):
        video_name = get_video_name(input_video_path)

        max_ = 0

        # Information for FPS
        start_time = time.time()
        FPS = 0.0
        completed = current_idx
        last = 0
        q_len = len(output_queue[0])

        while True:
            outputs = []

            try:
                img = frame_queue.get()
            except QueueClosedError:
                break

            for oq in output_queue:
                outputs.append(oq[completed % q_len].get())

            ## Make output image from predictions
            for postproc, (predictions, contexts, _) in zip(
                self.postprocessor, outputs
            ):
                img = postproc(predictions, contexts, img)

            if self.draw_fps:
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.3:
                    FPS = (completed - last) / elapsed_time
                    start_time = time.time()
                    last = completed
                h, w, c = img.shape
                dh = int(h * 0.1)
                dummy_img = np.zeros((h + dh, w, c)).astype(np.uint8)
                dummy_img[dh:, :, :] = img
                img = put_fps_to_img(dummy_img, FPS)
            result_queue.put(img)

            completed += self.num_handler_process
            cidx.value += 1
        
        result_queue.put(QueueStopEle)
        return


def get_video_name(video_path: str) -> str:
    return (video_path.split("/")[-1]).split(".")[0]


def put_fps_to_img(output_img: np.ndarray, FPS: float):
    h, w, _ = output_img.shape
    org = (int(0.01 * h), int(0.05 * w) - 5)
    scale = min(int(org[1] * 0.1), 3)
    output_img = cv2.putText(
        output_img,
        f"FPS: {FPS:.2f}",
        org,
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 255, 255),
        scale,
        cv2.LINE_AA,
    )
    return output_img
