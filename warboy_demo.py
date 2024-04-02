import os
import subprocess
import time
import multiprocessing as mp
import threading

import cv2
import psutil
import typer
from utils_.parse_params import get_demo_params_from_cfg
from utils_.mp_queue import MpQueue, QueueStopEle
from utils_.preprocess import YOLOPreProcessor
from utils_.postprocess import getPostProcesser
from utils_.handler import InputHandler, OutputHandler
from utils_.warboy_runner import WarboyRunner

app = typer.Typer(pretty_exceptions_show_locals=False)

class AppRunner:
    def __init__(self, param):
        self.app_type = param["app"]
        self.video_paths = param["video_paths"]
        self.runtime_params = param["runtime_params"]
        
        self.input_queue = MpQueue(5000)
        self.output_queues = [MpQueue(5000) for _ in range(len(self.video_paths))]

        self.furiosa_runtime = WarboyRunner(param["model_path"], param["worker_num"], param["warboy_device"])
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = getPostProcesser(self.app_type, param["model_name"], self.runtime_params, param["class_names"])

        self.input_handler = InputHandler(self.video_paths, param["output_path"], self.input_queue, self.preprocessor, param["input_shape"])
        self.output_handler = OutputHandler(self.video_paths, param["output_path"], self.output_queues, self.postprocessor, draw_fps = True)

    def __call__(self):
        warboy_runtime_process = mp.Process(target=self.furiosa_runtime, args=(self.input_queue, self.output_queues))
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
    def __init__(self, cfg, viewer="fastAPI"):
        self.demo_params = get_demo_params_from_cfg(cfg)
        self.viewer = viewer
        
        self.app_runners = [AppRunner(param) for param in self.demo_params]
        self.app_threads = [threading.Thread(target=app_runner, args=()) for app_runner in self.app_runners]

    def run(
        self,
    ):
        for app_thread in self.app_threads:
            app_thread.start()

        for app_thread in self.app_threads:
            app_thread.join()
        return

    def get_output_paths(self):
        output_paths = []
        for param in self.demo_params:
            for video_path in param["video_paths"]:
                video_name =  (video_path.split('/')[-1]).split('.')[0]
                output_paths.append(os.path.join(param["output_path"], "_output", video_name))
        return output_paths
    

@app.command()
def main(cfg, viewer):
    demo_app = DemoApplication(cfg, viewer)
    demo_app.run()

if __name__ == "__main__":
    app()
