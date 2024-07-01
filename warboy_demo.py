import asyncio
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time
from multiprocessing.managers import DictProxy, ListProxy, SyncManager

import cv2
import psutil
import typer
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.handler import InputHandler, OutputHandler
from utils.mp_queue import MpQueue, QueueStopEle
from utils.parse_params import get_demo_params_from_cfg
from utils.postprocess import getPostProcesser
from utils.preprocess import YOLOPreProcessor
from utils.warboy import WarboyDevice, WarboyRunner

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"))
warboy_device = None

sync_manager = SyncManager(("127.0.0.1", 5001), authkey=b"password")


def start_sync_manager(sync_manager):
    def run_sync_manager():
        result_queues = list()
        grid_info = {"num_channel": 1, "num_rows": 1, "num_cols": 1}

        def get_result_queues():
            return result_queues

        def get_grid_info():
            return grid_info

        SyncManager.register("get_result_queues", get_result_queues, ListProxy)
        SyncManager.register("get_grid_info", get_grid_info, DictProxy)
        sync_manager.get_server().serve_forever()

    proc = mp.Process(target=run_sync_manager)
    proc.start()
    return proc


class AppRunner:
    def __init__(self, param, result_queues):
        self.app_type = param["app"]
        self.videos_info = param["videos_info"]
        self.runtime_params = param["runtime_params"]
        self.input_queues = [
            [MpQueue(25) for _ in range(len(self.app_type))]
            for _ in range(len(self.videos_info))
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
        self.param = param

    def __call__(self):

        self.input_handler = InputHandler(
            self.videos_info,
            self.input_queues,
            self.frame_queues,
            self.preprocessor,
            self.param["input_shape"],
        )
        self.output_handler = OutputHandler(
            self.videos_info,
            self.output_queues,
            self.frame_queues,
            self.result_queues,
            self.postprocessor,
            draw_fps=True,
        )

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
        self.model_names = [
            [model_name for model_name in demo_param["model_name"]]
            for demo_param in self.demo_params
        ]

        SyncManager.register("get_result_queues")
        SyncManager.register("get_grid_info")
        mp.current_process().authkey = b"password"
        sync_manager.connect()

        for param in self.demo_params:
            app_result_queues = [
                sync_manager.Queue(8192) for _ in range(len(param["videos_info"]))
            ]
            self.app_runners.append(AppRunner(param, app_result_queues))
            self.result_queues += app_result_queues

        grid_info = sync_manager.get_grid_info()
        num_channel = sum([len(param["videos_info"]) for param in self.demo_params])
        grid_info.update(
            {
                "num_channel": num_channel,
                "num_rows": math.ceil(math.sqrt(num_channel)),
                "num_cols": math.ceil(math.sqrt(num_channel)),
            }
        )

        sync_manager.get_result_queues().extend(self.result_queues)

        self.app_threads = [
            threading.Thread(target=app_runner, args=())
            for app_runner in self.app_runners
        ]

    def run(
        self,
    ):

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
async def read_root(request: Request):
    grid_info = sync_manager.get_grid_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "num_rows": grid_info.get("num_rows"),
            "num_cols": grid_info.get("num_cols"),
            "num_channel": grid_info.get("num_channel"),
        },
    )


def getByteFrame(id):
    print(id)
    mp.current_process().authkey = b"password"
    result_queues = sync_manager.get_result_queues()
    result_queue = result_queues[id]
    while True:
        out_img = result_queue.get()
        out_img = out_img.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + out_img + b"\r\n")


@app.get("/video_feed/{id}")
async def stream(id):
    id = int(id)
    return StreamingResponse(
        getByteFrame(id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


async def generate_data():
    global warboy_device
    if warboy_device is None:
        warboy_device = await WarboyDevice.create()
    power, util, temp, se, devices = await warboy_device()
    return jsonable_encoder(
        {"power": power, "util": util, "temp": temp, "time": se, "devices": devices}
    )


@app.get("/chart_data")
async def get_data():
    t1 = time.time()
    datas = await generate_data()
    t2 = time.time()
    await asyncio.sleep(1 - (t2 - t1))
    return JSONResponse(content=datas)


@app.on_event("startup")
async def startup_event():
    SyncManager.register("get_result_queues")
    SyncManager.register("get_grid_info")
    mp.current_process().authkey = b"password"
    sync_manager.connect()


def run_web_server(port):
    uvicorn.run("warboy_demo:app", host="0.0.0.0", port=int(port))


def spawn_web_server(port):
    proc = mp.Process(target=run_web_server, args=(port,))
    proc.start()
    return proc


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "A config file path is only needed! (e.g., python warboy_demo.py [config_file])"

    proc = start_sync_manager(sync_manager)
    demo_params, port = get_demo_params_from_cfg(sys.argv[1])
    demo_app = DemoApplication(demo_params)
    run_demo_thread(demo_app)
    run_web_server(port)
    proc.join()

    print("Warboy Application End!!!!")
