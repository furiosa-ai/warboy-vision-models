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
from utils.result_img_process import ImageMerger
from utils.warboy import WarboyDevice, WarboyRunner

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"))
warboy_device = None


class AppRunner:
    def __init__(self, param, result_queues):
        self.app_type = param["app"]
        self.video_paths = param["video_paths"]
        self.runtime_params = param["runtime_params"]
        self.input_queues = [
            [MpQueue(25) for _ in range(len(self.app_type))]
            for _ in range(len(self.video_paths))
        ]
        self.frame_queues = [MpQueue(50) for _ in range(len(self.video_paths))]
        self.output_queues = [
            [[MpQueue(50) for _ in range(5)] for _ in range(len(self.app_type))]
            for _ in range(len(self.video_paths))
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
            self.video_paths,
            self.input_queues,
            self.frame_queues,
            self.preprocessor,
            param["input_shape"],
        )
        self.output_handler = OutputHandler(
            self.video_paths,
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
        self.input_handler.join()
        for input_q in self.input_queues:
            for iq in input_q:
                iq.put(QueueStopEle)

        for frame_queue in self.frame_queues:
            frame_queue.put(QueueStopEle)

        for warboy_runtime_process in warboy_runtime_processes:
            warboy_runtime_process.join()
        for output_queue in self.output_queues:
            for oq in output_queue:
                for o in oq:
                    o.put(QueueStopEle)
        self.output_handler.join()

        for result_queue in self.result_queues:
            result_queue.put(QueueStopEle)

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
        for param in self.demo_params:
            app_result_queues = [MpQueue(50) for _ in range(len(param["video_paths"]))]
            self.app_runners.append(AppRunner(param, app_result_queues))
            self.result_queues += app_result_queues

        self.app_threads = [
            threading.Thread(target=app_runner, args=())
            for app_runner in self.app_runners
        ]

    def run(
        self,
    ):
        merge_proc = mp.Process(
            target=self.merger,
            args=(
                self.model_names,
                self.result_queues,
            ),
        )
        for app_thread in self.app_threads:
            app_thread.start()
        merge_proc.start()
        for app_thread in self.app_threads:
            app_thread.join()
        merge_proc.join()
        return


def run_demo_thread(demo_application):
    t = threading.Thread(target=demo_application.run)
    t.daemon = True
    t.start()
    return t.native_id


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def getByteFrame():
    cnt = 0
    while True:
        img_path = os.path.join(".tmp", "%010d.bmp" % cnt)
        if not os.path.exists(img_path):
            continue
        out_img = cv2.imread(img_path)
        ret, out_img = cv2.imencode(".jpg", out_img)
        out_frame = out_img.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n"
        )
        os.remove(img_path)
        cnt += 1


@app.get("/video_feed")
async def stream():
    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
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
    d = await generate_data()
    t2 = time.time()
    await asyncio.sleep(1 - (t2 - t1))
    return JSONResponse(content=d)


def inside_func(*args, **kwargs):
    try:
        uvicorn.run(*args, **kwargs)
    except Exception as e:
        print(e, flush=True)
        with open("a.txt", "w") as f:
            print(e, file=f)


def run_web_server(port):
    proc = mp.Process(
        target=inside_func,
        args=("warboy_demo:app",),
        kwargs={
            "host": "0.0.0.0",
            "port": int(port),
        },
    )
    proc.start()
    return proc


def shutdown_web_server(proc):
    pid = proc.pid
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    proc.terminate()
    for _ in range(5):
        if proc.is_alive():
            time.sleep(1)
            print("Alive Web Server..")

    if proc.is_alive():
        subprocess.run(["kill", "-9", str(pid)])
    return


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "A config file path is only needed! (e.g., python warboy_demo.py [config_file])"

    if os.path.exists(".tmp"):
        subprocess.run(["rm", "-rf", ".tmp"])
    os.makedirs(".tmp")

    demo_params, port = get_demo_params_from_cfg(sys.argv[1])
    demo_app = DemoApplication(demo_params)
    run_demo_thread(demo_app)
    proc = run_web_server(port)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        subprocess.run(["rm", "-rf", ".tmp"])
        shutdown_web_server(proc)
        pass
    subprocess.run(["rm", "-rf", ".tmp"])
    print("EXIT!!!!")
