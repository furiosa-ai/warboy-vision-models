import threading
import multiprocessing as mp

import cv2
import time
import math
import queue
import numpy as np

from typing import List, Dict, Any, Tuple
from warboy.runtime.mp_queue import MpQueue, QueueClosedError, QueueStopEle
from warboy.utils.postprocess import getPostProcesser
from warboy.utils.preprocess import YOLOPreProcessor


class Handler:
    """

    """

    def __init__(
        self,
        input_queues: List[List[MpQueue]],
        output_queues: List[List[MpQueue]],
        result_queues: List[MpQueue],
        param: Dict[str, Any],
        num_channels: int,
    ) -> None:
        self.video_processors = [
            mp.Process(
                target=self.video_task,
                args=(video_info, input_queue, output_queue, result_queue),
            )
            for (video_info, input_queue, output_queue, result_queue) in zip(
                param["videos_info"], input_queues, output_queues, result_queues
            )
        ]
        self.input_shapes = param["input_shapes"]
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = [
            getPostProcesser(task, model_name, runtime_params, class_names)
            for task, model_name, runtime_params, class_names in zip(
                param["task"],
                param["model_names"],
                param["model_params"],
                param["class_names"],
            )
        ]
        self.full_grid_shape = (720, 1280)
        self.num_grid = None
        self.pad = 10
        self.grid_shape = self._get_grid_info(num_channels)

    def start(self):
        for proc in self.video_processors:
            try:
                proc.start()
            except Exception as e:
                print(e, flush=True)

    def join(self):
        for proc in self.video_processors:
            proc.join()

    def video_task(
        self,
        video_info: Dict[str, Any],
        input_queue: List[MpQueue],
        output_queue: List[MpQueue],
        result_queue: MpQueue,
    ) -> None:
        """

        """
        video_path, video_type, recursive = video_info.values()
        running = True
        img_idx = 0
        num_completed = 0
        FPS = 0.0
        start_time = time.time()
        cap = self._get_cv_widget(video_path, video_type)
        while True:
            try:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    if recursive:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                contexts = self._put_input_to_queue(frame, input_queue)
                out_img = self._get_output_from_queue(frame, output_queue, contexts)
                if out_img is None:
                    break

                if time.time() - start_time > 1.0:
                    FPS = (img_idx - num_completed) / (time.time() - start_time)
                    start_time = time.time()
                    num_completed = img_idx

                out_img = self._put_fps_to_img(out_img, f"FPS: {FPS:.1f}")
                out_img = cv2.resize(
                    out_img, self.grid_shape, interpolation=cv2.INTER_LINEAR
                )
                result_queue.put((out_img, FPS))
                img_idx += 1
            except Exception as e:
                result_queue.put(QueueStopEle)
                break
        if cap.isOpened():
            cap.release()

    def _put_input_to_queue(self, frame: np.ndarray, input_queue: List[MpQueue]):
        contexts = []
        for input_shape, iq in zip(self.input_shapes, input_queue):
            input_, context = self.preprocessor(frame, input_shape)
            contexts.append(context)
            iq.put(input_)
        return contexts

    def _get_output_from_queue(
        self,
        frame: np.ndarray,
        output_queue: List[MpQueue],
        contexts: List[Dict[str, Any]],
    ):
        out_img = frame
        for postprocess, oq, context in zip(self.postprocessor, output_queue, contexts):
            t0 = time.time()
            while True:
                try:
                    outputs = oq.get(False)
                    out_img = postprocess(outputs, context, out_img)
                    break
                except queue.Empty:
                    time.sleep(1e-6)
                except QueueClosedError:
                    return None
        return out_img

    def _put_fps_to_img(self, img, FPS):
        h, w, _ = img.shape
        c1 = (int(0.01 * h) + 2, int(0.05 * w) + 5)
        tl = min(int(c1[1] * 0.1), 3)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(FPS, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + int(t_size[0] * 1.5), c1[1] - t_size[1] - int(0.05 * h) + 5
        # cv2.rectangle(img, (c1[0], c1[1] + 5), c2, (0, 0, 0), -1)  # filled
        cv2.putText(
            img,
            FPS,
            (c1[0], c1[1] + 2),
            cv2.FONT_HERSHEY_PLAIN,
            tl,
            (0, 0, 0),
            thickness=tl,
            lineType=cv2.LINE_AA,
        )
        return img

    def _get_cv_widget(self, video_path: str, video_type: str):
        cap = None
        if video_type == "file":
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        elif video_type == "cctv":
            pass
        elif video_type == "webcam":
            pass
        else:
            raise "Not Supported Video Type ()"
        return cap

    def _get_grid_info(self, num_channel):
        if self.num_grid is None:
            n = math.ceil(math.sqrt(num_channel))
            self.num_grid = (n, n)
        grid_shape = (
            int((self.full_grid_shape[1]) / self.num_grid[1]) - self.pad // 2,
            int((self.full_grid_shape[0]) / self.num_grid[0]) - self.pad // 2,
        )
        return grid_shape


class ImageHandler:
    """

    """

    def __init__(
        self,
        full_grid_shape: Tuple[int, int] = (720, 1280),
        num_grid: Tuple[int, int] = None,
    ):
        self.full_grid_shape = full_grid_shape
        self.num_grid = num_grid  # row, column
        self.pad = 10

    def __call__(self, result_queues: List[MpQueue]):
        num_channel = len(result_queues)
        num_end_channel = 0

        grid_shape = self._get_grid_info(num_channel)
        while num_end_channel < num_channel:
            idx = 0
            grid_imgs = []
            total_fps = 0
            while idx < num_channel:
                out_img = None
                while True:
                    try:
                        out_img, FPS = result_queues[idx].get(False)
                        break
                    except queue.Empty:
                        time.sleep(1e-6)
                    except QueueClosedError:
                        num_end_channel += 1 if states[idx] else 0
                        states[idx] = False
                        break
                grid_imgs.append(out_img)
                total_fps += FPS
                idx += 1
            full_grid_img = self._get_full_grid_img(
                grid_imgs, grid_shape, total_fps, num_channel - num_end_channel
            )
            yield full_grid_img, total_fps

    def _get_grid_info(self, num_channel):
        if self.num_grid is None:
            n = math.ceil(math.sqrt(num_channel))
            self.num_grid = (n, n)
        grid_shape = (
            int((self.full_grid_shape[1]) / self.num_grid[1]) - self.pad // 2,
            int((self.full_grid_shape[0]) / self.num_grid[0]) - self.pad // 2,
        )
        return grid_shape

    def _get_full_grid_img(self, grid_imgs, grid_shape, total_fps=0.0, num_videos=0):
        height_pad = 0  # int(self.full_grid_shape[0] * 0.06)

        full_grid_img = np.zeros(
            (self.full_grid_shape[0] + height_pad, self.full_grid_shape[1], 3), np.uint8
        )
        for i, grid_img in enumerate(grid_imgs):
            if grid_img is None:
                continue

            c = i % self.num_grid[0]
            r = i // self.num_grid[0]

            x0 = c * grid_shape[1] + (c + 1) * (self.pad // 2) + height_pad
            x1 = (c + 1) * grid_shape[1] + (c + 1) * (self.pad // 2) + height_pad

            y0 = r * grid_shape[0] + (r + 1) * (self.pad // 2)
            y1 = (r + 1) * grid_shape[0] + (r + 1) * (self.pad // 2)

            full_grid_img[x0:x1, y0:y1] = grid_img
        return full_grid_img
