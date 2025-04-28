import math
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from ..face_recognition.preprocess import FaceRecogPreProcessor
from ..runtime.warboy_runtime import WarboyApplication, WarboyQueueRuntime
from ..yolo.postprocess import get_post_processor
from ..yolo.preprocess import YoloPreProcessor
from .image_decoder import ImageListDecoder
from .image_encoder import ImageEncoder, PredictionEncoder
from .queue import PipeLineQueue, QueueClosedError
from .video_decoder import VideoDecoder


@dataclass
class Engine:
    name: str
    task: str
    model: str
    model_type: str = "yolov8n"
    worker_num: int = 8
    device: str = "warboy(2)*1"
    class_names: List[str] = None
    conf_thres: float = 0.25
    iou_thres: float = 0.7
    input_shape: Tuple[int, int] = (640, 640)
    anchors = [None]
    use_tracking: bool = True

    def _get_runtime_info(self):
        return

    def _get_func_info(self):
        return


@dataclass
class Video:
    video_info: str
    recursive: bool


@dataclass
class Image:
    image_info: str


@dataclass
class ImageList:
    image_list: List[Image]


class PipeLine:
    def __init__(
        self,
        num_channels: int,
        run_fast_api: bool = True,
        run_e2e_test: bool = False,
        make_image_output: bool = False,
    ):
        self.run_fast_api = run_fast_api
        self.run_e2e_test = run_e2e_test
        self.make_image_output = make_image_output
        self.runtime_info = {}
        self.preprocess_functions = {}
        self.postprocess_functions = {}

        self.video_decoder_process = []
        self.image_encoder_process = []

        self.stream_mux_list = defaultdict(list)
        self.frame_mux_list = defaultdict(list)
        self.output_mux_list = defaultdict(list)
        self.result_mux_list = defaultdict(list)

        self.outputs = {}
        self.image_paths_dict = {}
        self.image_paths = []

        # for resize... use ImageHandler
        self.image_handler = ImageHandler(num_channels=num_channels)

        self.results = []

    def add(self, obj, name: str = "", postprocess_as_img=True):
        if isinstance(obj, Engine):
            self.runtime_info[obj.name] = {
                "model": obj.model,
                "worker_num": obj.worker_num,
                "device": obj.device,
            }
            if "yolo" in obj.model:
                self.preprocess_functions[obj.name] = YoloPreProcessor(
                    new_shape=obj.input_shape, tensor_type="uint8"
                )
            elif "face_recognition" == obj.task:
                self.preprocess_functions[obj.name] = FaceRecogPreProcessor(
                    new_shape=obj.input_shape, tensor_type="uint8"
                )
            else:
                raise "Error: not implemented model type"
            if postprocess_as_img:
                self.postprocess_functions[obj.name] = get_post_processor(
                    task=obj.task,
                    model_name=obj.model_type,
                    model_cfg={
                        "conf_thres": obj.conf_thres,
                        "iou_thres": obj.iou_thres,
                        "anchors": obj.anchors,
                    },
                    class_names=obj.class_names,
                    use_trakcing=obj.use_tracking,
                )
            else:
                self.postprocess_functions[obj.name] = get_post_processor(
                    task=obj.task,
                    model_name=obj.model_type,
                    model_cfg={
                        "conf_thres": obj.conf_thres,
                        "iou_thres": obj.iou_thres,
                        "anchors": obj.anchors,
                    },
                    class_names=obj.class_names,
                    use_trakcing=obj.use_tracking,
                ).postprocess_func
        elif isinstance(obj, Video):
            if not name in self.runtime_info:
                raise "Error"

            new_stream_mux = PipeLineQueue(maxsize=500)
            new_frame_mux = PipeLineQueue(maxsize=500)
            new_output_mux = PipeLineQueue(maxsize=500)
            new_result_mux = (
                PipeLineQueue(maxsize=500)
                if (self.run_fast_api or self.run_e2e_test or self.make_image_output)
                else None
            )
            self.stream_mux_list[name].append(new_stream_mux)
            self.frame_mux_list[name].append(new_frame_mux)
            self.output_mux_list[name].append(new_output_mux)
            self.result_mux_list[name].append(new_result_mux)

            self.video_decoder_process.append(
                VideoDecoder(
                    video_path=obj.video_info,
                    stream_mux=new_stream_mux,
                    frame_mux=new_frame_mux,
                    preprocess_function=self.preprocess_functions[name],
                    recursive=obj.recursive,
                )
            )
            self.image_encoder_process.append(
                ImageEncoder(
                    frame_mux=new_frame_mux,
                    output_mux=new_output_mux,
                    result_mux=new_result_mux,
                    postprocess_function=self.postprocess_functions[name],
                )
            )
        elif isinstance(obj, ImageList):
            if not name in self.runtime_info:
                raise "Error"

            new_stream_mux = PipeLineQueue(maxsize=500)
            new_frame_mux = PipeLineQueue(maxsize=500)
            new_output_mux = PipeLineQueue(maxsize=500)
            new_result_mux = (
                PipeLineQueue(maxsize=500)
                if (self.run_fast_api or self.run_e2e_test or self.make_image_output)
                else None
            )
            self.stream_mux_list[name].append(new_stream_mux)
            self.frame_mux_list[name].append(new_frame_mux)
            self.output_mux_list[name].append(new_output_mux)
            self.result_mux_list[name].append(new_result_mux)
            self.image_paths_dict[name] = [image.image_info for image in obj.image_list]
            # Preprocess
            self.video_decoder_process.append(
                ImageListDecoder(
                    obj.image_list,
                    stream_mux=new_stream_mux,
                    frame_mux=new_frame_mux,
                    preprocess_function=self.preprocess_functions[name],
                )
            )
            # Postprocess (draw bbox for object detection)
            if postprocess_as_img:
                self.image_encoder_process.append(
                    ImageEncoder(
                        frame_mux=new_frame_mux,
                        output_mux=new_output_mux,
                        result_mux=new_result_mux,
                        postprocess_function=self.postprocess_functions[name],
                    )
                )
            else:
                self.image_encoder_process.append(
                    PredictionEncoder(
                        frame_mux=new_frame_mux,
                        output_mux=new_output_mux,
                        result_mux=new_result_mux,
                        postprocess_function=self.postprocess_functions[name],
                    )
                )
        else:
            raise "Error: not implemented type"

    def run(self, runtime_type: str = "queue"):
        if runtime_type == "queue":
            runtime_process = [
                WarboyQueueRuntime(
                    model=runtime_info["model"],
                    worker_num=runtime_info["worker_num"],
                    device=runtime_info["device"],
                    stream_mux_list=self.stream_mux_list[name],
                    output_mux_list=self.output_mux_list[name],
                )
                for name, runtime_info in self.runtime_info.items()
            ]
        elif runtime_type == "application":
            runtime_process = [
                WarboyApplication(
                    model=runtime_info["model"],
                    worker_num=runtime_info["worker_num"],
                    device=runtime_info["device"],
                    stream_mux_list=self.stream_mux_list[name],
                    output_mux_list=self.output_mux_list[name],
                )
                for name, runtime_info in self.runtime_info.items()
            ]
        else:
            raise "Error: runtime_type must be queue or application"

        try:
            pipeline_procs = []
            pipeline_procs += [
                mp.Process(target=proc.run, args=(), name=str(proc))
                for proc in self.video_decoder_process
            ]
            pipeline_procs += [
                mp.Process(target=proc.run, args=(), name=str(proc))
                for proc in runtime_process
            ]
            pipeline_procs += [
                mp.Process(target=proc.run, args=(), name=str(proc))
                for proc in self.image_encoder_process
            ]
            for pipeline_proc in pipeline_procs:
                pipeline_proc.start()

            if self.run_fast_api:
                total_result_mux_list = [
                    self.result_mux_list[name] for name, _ in self.runtime_info.items()
                ]
                self.image_handler.output_stream_handler(
                    [
                        result_mux
                        for result_mux_list in total_result_mux_list
                        for result_mux in result_mux_list
                    ]
                )
            elif self.run_e2e_test:
                total_result_mux_list = [
                    (name, self.result_mux_list[name])
                    for name, _ in self.runtime_info.items()
                ]
                self.image_handler.output_e2e_test_handler(
                    self.outputs,
                    [
                        (name, result_mux)
                        for (name, result_mux_list) in total_result_mux_list
                        for result_mux in result_mux_list
                    ],
                    self.image_paths_dict,
                    self.image_paths,
                )
            elif self.make_image_output:
                total_result_mux_list = [
                    self.result_mux_list[name] for name, _ in self.runtime_info.items()
                ]
                self.image_handler.output_image_handler(
                    [
                        result_mux
                        for result_mux_list in total_result_mux_list
                        for result_mux in result_mux_list
                    ]
                )

            for pipeline_proc in pipeline_procs:
                pipeline_proc.join()
        except Exception as e:
            print(e)
            pass


class ImageHandler:
    def __init__(self, num_channels):
        self.full_grid_shape = (720, 1280)
        self.num_grid = None
        self.pad = 10
        self.grid_shape = self._get_grid_info(num_channels)

    def _get_grid_info(self, num_channels):
        if self.num_grid is None:
            n = math.ceil(math.sqrt(num_channels))
            self.num_grid = (n, n)
        grid_shape = (
            int((self.full_grid_shape[1]) / self.num_grid[1]) - self.pad // 2,
            int((self.full_grid_shape[0]) / self.num_grid[0]) - self.pad // 2,
        )
        return grid_shape

    def _get_full_grid_img(self, grid_imgs, grid_shape):
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

    def _put_fps_to_img(self, img, FPS):
        h, w, _ = img.shape
        c1 = (int(0.01 * h) + 2, int(0.05 * w) + 5)
        tl = min(int(c1[1] * 0.1), 3)
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

    def output_stream_handler(self, result_mux_list: List[PipeLineQueue]):
        end_channels = 0
        id_ = 0
        t1 = time.time()

        while True:
            grid_imgs = []
            total_fps = 0
            for result_mux in result_mux_list:
                if result_mux == None:
                    continue
                try:
                    # obj detection, output = bboxed image
                    output, fps, _ = result_mux.get()
                    output_img = self._put_fps_to_img(output, f"FPS: {fps:.1f}")
                    output_img = cv2.resize(
                        output_img, self.grid_shape, interpolation=cv2.INTER_NEAREST
                    )
                    total_fps += fps
                    grid_imgs.append(output_img)
                except QueueClosedError:
                    end_channels += 1
                    if end_channels == len(result_mux_list):
                        break
                    continue

            full_grid_img = self._get_full_grid_img(grid_imgs, self.grid_shape)
            yield full_grid_img, total_fps

            id_ += 1

            if end_channels == len(result_mux_list):
                break

    def output_image_handler(self, result_mux_list: List[PipeLineQueue]):
        end_channels = 0
        id_ = 0

        if not os.path.exists("./outputs"):
            os.makedirs("./outputs")

        while True:
            for idx, result_mux in enumerate(result_mux_list):
                if not os.path.exists(f"./outputs/img{idx}"):
                    os.makedirs(f"./outputs/img{idx}")
                if result_mux == None:
                    continue
                try:
                    # obj detection, output = bboxed image
                    output, fps, _ = result_mux.get()
                    output_img = cv2.resize(
                        output, self.grid_shape, interpolation=cv2.INTER_NEAREST
                    )
                    cv2.imwrite(f"./outputs/img{idx}/{id_}.jpg", output_img)
                except QueueClosedError:
                    end_channels += 1
                    if end_channels == len(result_mux_list):
                        break
                    continue

            id_ += 1

            if end_channels == len(result_mux_list):
                break

    def output_e2e_test_handler(
        self,
        outputs,
        result_mux_list: List[PipeLineQueue],
        image_paths_dict,
        image_paths,
    ):
        end_channels = 0

        while True:
            for name, result_mux in result_mux_list:
                if result_mux == None:
                    continue
                try:
                    output, _, img_idx = result_mux.get()
                    if not len(output) == 1:
                        print(len(output))
                    image_path = image_paths_dict[name][img_idx]
                    if image_path not in outputs:
                        outputs[image_path] = [output[0]]
                    else:
                        outputs[image_path].append(output[0])
                    image_paths.append(image_paths_dict[name][img_idx])
                except QueueClosedError:
                    end_channels += 1
                    if end_channels == len(result_mux_list):
                        break
                    continue

            if end_channels == len(result_mux_list):
                break
