import asyncio
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import threading
import time

import cv2
from furiosa import runtime
import typer
import uvloop
import yaml

from utils.mp_queue import *
from utils.postprocess import ObjDetPostProcess, PoseDetPostProcess
from utils.preprocess import YOLOPreProcessor, letterbox
from video_utils.output_proc import OutProcessor
from video_utils.video_proc import VideoPreProcessor
from video_utils.viewer import WarboyViewer

app = typer.Typer(pretty_exceptions_show_locals=False)


class FuriosaApplication:
    def __init__(self, model_path, worker_num, device):
        self.model_path = model_path
        self.worker_num = worker_num
        self.device = device

    def __call__(self, inputQ, outputQs):
        asyncio.run(self.runner(inputQ, outputQs))

    async def runner(self, inputQ, outputQs):
        async with runtime.create_queue(
            model=self.model_path, worker_num=self.worker_num, device=self.device
        ) as (submitter, receiver):
            submit_task = asyncio.create_task(self.submit_with(submitter, inputQ))
            recv_task = asyncio.create_task(self.recv_with(receiver, outputQs))
            await submit_task
            await recv_task

    async def submit_with(self, submitter, inputQ):
        while True:
            try:
                input_, preproc_params, img_idx, video_idx = inputQ.get()
            except QueueClosedError:
                break
            await submitter.submit(input_, context=(preproc_params, img_idx, video_idx))

    async def recv_with(self, receiver, outputQs):
        while True:

            async def recv():
                context, outputs = await receiver.recv()
                return context, outputs

            try:
                recv_task = asyncio.create_task(recv())
                (preproc_params, img_idx, video_idx), outputs = await asyncio.wait_for(
                    recv_task, timeout=1
                )
            except asyncio.TimeoutError:
                break

            try:
                outputQs[video_idx].put((outputs, preproc_params, img_idx))
            except queue.Full:
                time.sleep(0.01)
                outputQs[video_idx].put((outputs, preproc_params, img_idx))


def set_processor(app_type, model_name, runner_info):
    pre_processor = None
    post_processor = None
    img_to_img = False

    if app_type == "detection":
        pre_processor = YOLOPreProcessor()
        post_processor = ObjDetPostProcess(model_name, runner_info)
    elif app_type == "pose":
        pre_processor = YOLOPreProcessor()
        post_processor = PoseDetPostProcess(model_name, runner_info)
    elif app_type == "tracking":
        pass
    elif app_type == "segmentation":
        pass
    else:
        raise "Unsupported Application!"

    return pre_processor, post_processor, img_to_img


def app_runner(param):
    video_paths = param["video_paths"]
    inputQ = MpQueue()
    outputQs = [MpQueue() for _ in range(len(video_paths))]
    output_path = param["output_path"]

    furiosa_app = FuriosaApplication(
        param["model_path"], param["worker_num"], param["warboy_device"]
    )
    pre_processor, post_processor, img_to_img = set_processor(
        param["app"], param["model_name"], param["runner_info"]
    )

    ##### Process Setting #####
    draw_fps = True
    video_proc = VideoPreProcessor(video_paths, output_path, pre_processor, inputQ, img_to_img)
    output_proc = OutProcessor(
        video_paths, output_path, post_processor, outputQs, draw_fps, img_to_img
    )
    furiosa_proc = mp.Process(target=furiosa_app, args=(inputQ, outputQs))

    ##### Processes Run #####
    app_start_time = time.time()
    video_proc.start()
    furiosa_proc.start()
    output_proc.start()

    video_proc.join()
    inputQ.put(QueueStopEle)
    furiosa_proc.join()
    for outputQ in outputQs:
        outputQ.put(QueueStopEle)
    output_proc.join()

    app_end_time = time.time()
    app = param["app"]
    print(f"Application: {app} -> Time: {app_end_time-app_start_time}s")


def get_params_from_cfg(cfg: str):
    num_channel = 0
    with open(cfg) as f:
        app_infos = yaml.load_all(f, Loader=yaml.FullLoader)
        params = []
        for app_info in app_infos:
            model_config = open(app_info["model_config"])
            model_info = yaml.load(model_config, Loader=yaml.FullLoader)
            model_config.close()

            if os.path.exists(app_info["output_path"]):
                subprocess.run(["rm", "-rf", app_info["output_path"]])
            os.makedirs(app_info["output_path"])

            params.append(
                {
                    "app": app_info["app"],
                    "runner_info": model_info["runner_info"],
                    "model_name": app_info["model_name"],
                    "model_path": app_info["model_path"],
                    "worker_num": int(app_info["worker_num"]),
                    "warboy_device": app_info["device"],
                    "video_paths": app_info["video_path"],
                    "output_path": app_info["output_path"],
                }
            )

    return params


@app.command()
def run_demo(cfg):
    params = get_params_from_cfg(cfg)
    app_threads = []

    #warboy_viewer = WarboyViewer()

    for param in params:
        app_thread = threading.Thread(target=app_runner, args=(param,))
        app_threads.append(app_thread)
        app_thread.start()

    #warboy_viewer.start()

    for app_thread in app_threads:
        app_thread.join()

    #warboy_viewer.state = False
    #warboy_viewer.join()
    return


if __name__ == "__main__":
    uvloop.install()
    app()
