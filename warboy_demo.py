import multiprocessing as mp
import os
import subprocess
import threading
import time

import cv2
import psutil
import typer

from utils.handler import InputHandler, OutputHandler
from utils.mp_queue import MpQueue, QueueStopEle
from utils.parse_params import get_demo_params_from_cfg
from utils.postprocess import getPostProcesser
from utils.preprocess import YOLOPreProcessor
from utils.warboy_runner import WarboyRunner

app = typer.Typer(pretty_exceptions_show_locals=False)


class AppRunner:
    def __init__(self, param):
        self.app_type = param["app"]
        self.video_paths = param["video_paths"]
        self.runtime_params = param["runtime_params"]

        self.input_queue = MpQueue(5000)
        self.output_queues = [MpQueue(5000) for _ in range(len(self.video_paths))]

        self.furiosa_runtime = WarboyRunner(
            param["model_path"], param["worker_num"], param["warboy_device"]
        )
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = getPostProcesser(
            self.app_type,
            param["model_name"],
            self.runtime_params,
            param["class_names"],
        )

        self.input_handler = InputHandler(
            self.video_paths,
            param["output_path"],
            self.input_queue,
            self.preprocessor,
            param["input_shape"],
        )
        self.output_handler = OutputHandler(
            self.video_paths,
            param["output_path"],
            self.output_queues,
            self.postprocessor,
            draw_fps=True,
        )

    def __call__(self):
        warboy_runtime_process = mp.Process(
            target=self.furiosa_runtime, args=(self.input_queue, self.output_queues)
        )
        self.input_handler.start()
        warboy_runtime_process.start()
        self.output_handler.start()

        self.input_handler.join()
        self.input_queue.put(QueueStopEle)
        warboy_runtime_process.join()
        for output_queue in self.output_queues:
            output_queue.put(QueueStopEle)
        self.output_handler.join()

        print(f"Application -> {self.app_type} End!!")


class DemoApplication:
    def __init__(self, cfg, viewer=None):
        self.cfg = cfg
        self.demo_params = get_demo_params_from_cfg(cfg)
        self.app_runners = [AppRunner(param) for param in self.demo_params]
        self.app_threads = [
            threading.Thread(target=app_runner, args=())
            for app_runner in self.app_runners
        ]
        self.viewer = viewer

    def run(
        self,
    ):
        for app_thread in self.app_threads:
            app_thread.start()

        stream_process = None
        if self.viewer is not None:
            stream_process = subprocess.Popen(
                ["python", "tools/stream.py", self.cfg, self.viewer]
            )
        for app_thread in self.app_threads:
            app_thread.join()
        time.sleep(10)
        self.shutdown_proc(stream_process)
        return

    def shutdown_proc(self, proc):
        if proc is None:
            return

        pid = proc.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        proc.terminate()


@app.command()
def main(cfg, viewer):
    demo_app = DemoApplication(cfg, viewer)
    demo_app.run()


if __name__ == "__main__":
    app()
