import asyncio
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time

import cv2
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psutil
import typer
import uvicorn

from utils.handler import InputHandler, OutputHandler
from utils.mp_queue import MpQueue, QueueStopEle
from utils.parse_params import get_demo_params_from_cfg
from utils.postprocess import getPostProcesser
from utils.preprocess import YOLOPreProcessor
from utils.result_img_process import ImageMerger
from utils.warboy import WarboyDevice, WarboyRunner

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"))
warboy_device = None

app.state.model_names = list()
app.state.result_queues = list()


class AppRunner:
    def __init__(self, param, result_queues):
        self.app_type = param["app"]
        self.videos_info = param["videos_info"]
        self.runtime_params = param["runtime_params"]
        self.input_queues = [
            [MpQueue(25) for _ in range(len(self.app_type))] for _ in range(len(self.videos_info))
        ]
        self.frame_queues = [MpQueue(50) for _ in range(len(self.videos_info))]
        self.output_queues = [
            [[MpQueue(50) for _ in range(5)] for _ in range(len(self.app_type))]
            for _ in range(len(self.videos_info))
        ]
        self.result_queues = result_queues
        self.furiosa_runtimes = [
            WarboyRunner(
                param["model_path"][idx],
                param["worker_num"],
                param["warboy_device"][idx],
            )
            for idx in range(len(self.app_type))
        ]

        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = [
            getPostProcesser(
                self.app_type[idx],
                param["model_name"][idx],
                self.runtime_params[idx],
                param["class_names"][idx],
            )
            for idx in range(len(self.app_type))
        ]

        self.input_handler = InputHandler(
            self.videos_info,
            self.input_queues,
            self.frame_queues,
            self.preprocessor,
            param["input_shape"],
        )
        self.output_handler = OutputHandler(
            self.videos_info,
            self.output_queues,
            self.frame_queues,
            self.result_queues,
            self.postprocessor,
            draw_fps=True,
        )

    def __call__(self):
        warboy_runtime_processes = [
            mp.Process(
                target=furiosa_runtime,
                args=(
                    self.input_queues,
                    self.output_queues,
                    idx,
                ),
            )
            for idx, furiosa_runtime in enumerate(self.furiosa_runtimes)
        ]
        self.input_handler.start()
        for warboy_runtime_process in warboy_runtime_processes:
            warboy_runtime_process.start()
        self.output_handler.start()
        for warboy_runtime_process in warboy_runtime_processes:
            warboy_runtime_process.join()

        self.output_handler.join()

        print(f"Application -> {self.app_type} End!!")


class DemoApplication:
    def __init__(self, demo_params):
        self.demo_params = demo_params
        self.result_queues = []
        self.app_runners = []
        self.merger = ImageMerger()
        self.model_names = [
            [model_name for model_name in demo_param["model_name"]]
            for demo_param in self.demo_params
        ]

        manager = mp.Manager()
        for param in self.demo_params:
            app_result_queues = [manager.Queue(8192) for _ in range(len(param["videos_info"]))]
            self.app_runners.append(AppRunner(param, app_result_queues))
            self.result_queues += app_result_queues

        self.app_threads = [
            threading.Thread(target=app_runner, args=()) for app_runner in self.app_runners
        ]

    def run(
        self,
    ):
        app.state.model_names = self.model_names
        app.state.result_queues = self.result_queues

        for app_thread in self.app_threads:
            app_thread.start()
        for app_thread in self.app_threads:
            app_thread.join()
        return


def run_demo_thread(demo_application):
    t = threading.Thread(target=demo_application.run)
    t.daemon = True
    t.start()
    return t.native_id


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def getByteFrame():
    merger = ImageMerger()
    model_names = app.state.model_names
    result_queues = app.state.result_queues

    for out_img in merger(model_names, result_queues):
        ret, out_img = cv2.imencode(".jpg", out_img)
        out_frame = out_img.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n")


@app.get("/video_feed")
def stream():
    return StreamingResponse(getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame")


def generate_data():
    global warboy_device
    if warboy_device is None:
        warboy_device = asyncio.run(WarboyDevice.create())
    power, util, temp, se, devices = asyncio.run(warboy_device())
    return jsonable_encoder(
        {"power": power, "util": util, "temp": temp, "time": se, "devices": devices}
    )


@app.get("/chart_data")
def get_data():
    t1 = time.time()
    datas = generate_data()
    t2 = time.time()
    time.sleep(1 - (t2 - t1))
    return JSONResponse(content=datas)


def run_web_server():
    uvicorn.run(app, host="0.0.0.0", port=int(port))


def spawn_web_server(port):
    proc = mp.Process(
        target=run_web_server,
    )
    proc.start()
    return proc


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "A config file path is only needed! (e.g., python warboy_demo.py [config_file])"

    demo_params, port = get_demo_params_from_cfg(sys.argv[1])
    demo_app = DemoApplication(demo_params)
    run_demo_thread(demo_app)
    run_web_server()

    print("Warboy Application End!!!!")
