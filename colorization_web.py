import asyncio
import glob
import multiprocessing as mp
import os
import random
import shutil
import signal
import time
from concurrent.futures import (
    ALL_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import cv2
import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from furiosa.runtime import create_queue

from utils.result_img_process import ImageMerger
from utils.warboy import WarboyDevice

NUM_INPUT_WORKER = 8

OVERLAP = 0
PAD = 32
_SINGLE_PAD = PAD // 2

IN_PATCH_SIZE = 256
IN_STRIDE = IN_PATCH_SIZE - OVERLAP - PAD

OUT_PATCH_SIZE = IN_PATCH_SIZE - PAD
OUT_STRIDE = OUT_PATCH_SIZE - OVERLAP

NUM_WIDTH = 1
NUM_HEIGHT = 1

INPUT_HEIGHT = NUM_HEIGHT * IN_STRIDE + OVERLAP + PAD
INPUT_WIDTH = NUM_WIDTH * IN_STRIDE + OVERLAP + PAD
OUTPUT_HEIGHT = NUM_HEIGHT * OUT_STRIDE + OVERLAP
OUTPUT_WIDTH = NUM_WIDTH * OUT_STRIDE + OVERLAP

onnx_path = "colorization_after_1_best_i8_origin.onnx"

DATA_PATHS = glob.glob("video/*.wmv")

try:
    shutil.rmtree(".tmp")
    Path(".tmp").mkdir(parents=True, exist_ok=True)
except:
    pass


app = FastAPI()
warboy_device = WarboyDevice()
MERGER = ImageMerger()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def async_get_patch(video_Q, input_Q):
    out = video_Q.get()
    if not out:
        return out
    img, data_path = out
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    h, w, _ = img.shape
    for h_idx in range(NUM_HEIGHT):
        for w_idx in range(NUM_WIDTH):
            hh = h_idx * IN_STRIDE
            ww = w_idx * IN_STRIDE

            patch = img[hh : hh + IN_PATCH_SIZE, ww : ww + IN_PATCH_SIZE]
            patch = patch.transpose((2, 0, 1))[::-1].copy()
            patch = np.ascontiguousarray(np.expand_dims(patch, 0), dtype=np.uint8)
            input_Q.put((patch, h_idx, w_idx, data_path))

    return True


def gather_patch(img_mat):
    num_h = len(img_mat)
    num_w = len(img_mat[0])

    full_img = np.zeros(
        (
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
            3,
        )
    )

    for h_idx in range(num_h):
        for w_idx in range(num_w):
            out_patch = img_mat[h_idx][w_idx][
                _SINGLE_PAD : _SINGLE_PAD + OUT_PATCH_SIZE,
                _SINGLE_PAD : _SINGLE_PAD + OUT_PATCH_SIZE,
            ]
            hh = h_idx * OUT_STRIDE
            ww = w_idx * OUT_STRIDE
            full_img[hh : hh + OUT_PATCH_SIZE, ww : ww + OUT_PATCH_SIZE] = np.add(
                out_patch, full_img[hh : hh + OUT_PATCH_SIZE, ww : ww + OUT_PATCH_SIZE]
            )

            if h_idx == 0 and w_idx == 0:
                continue
            if h_idx > 0:
                full_img[hh : hh + OVERLAP, ww : ww + OUT_PATCH_SIZE] /= 2
            if w_idx > 0:
                full_img[hh : hh + OUT_PATCH_SIZE, ww : ww + OVERLAP] /= 2
            if h_idx > 0 and w_idx > 0:
                full_img[hh : hh + OVERLAP, ww : ww + OVERLAP] *= 2

    return full_img


def get_patch_wrapper(video_Q, patch_Q):
    while async_get_patch(video_Q, patch_Q):
        pass


async def async_create_patch(executor, loop, video_Q, patch_Q):
    print("START async_create_patch")

    exec_l = list()
    for _ in range(NUM_INPUT_WORKER):
        exec_l.append(
            loop.run_in_executor(executor, get_patch_wrapper, video_Q, patch_Q)
        )

    for t in exec_l:
        await t

    print("END preproc_task")
    patch_Q.put(None)
    return True


async def async_gather_patch(output_Q, result_Qs, reciever):
    print("START async_gather_patch")
    await asyncio.sleep(1)

    curr_idx = -1
    data_path = -1
    img_mat = dict()
    img_mat_count = dict()

    while True:
        if await output_Q.get() is None:
            return

        (y_idx, x_idx, data_path), patch_out = await reciever.recv()

        patch_out = patch_out[0][0]
        patch_out = patch_out[::-1].copy()
        patch_out = patch_out.astype(np.uint8)
        patch_out = patch_out.squeeze().transpose((1, 2, 0))

        if curr_idx == -1:
            curr_idx = data_path

        if data_path not in img_mat.keys():
            img_mat[data_path] = [
                [None for _ in range(NUM_WIDTH)] for _ in range(NUM_HEIGHT)
            ]
            img_mat_count[data_path] = 0

        img_mat[data_path][y_idx][x_idx] = patch_out
        img_mat_count[data_path] += 1

        if img_mat_count[data_path] >= NUM_HEIGHT * NUM_WIDTH:
            path_name = os.path.basename(data_path)
            path_name = os.path.splitext(path_name)[0]
            path_dir = os.path.dirname(data_path)
            path_dir = os.path.basename(path_dir)
            result_Qs[os.path.dirname(data_path)].put(gather_patch(img_mat[data_path]))
            del img_mat[data_path]
            del img_mat_count[data_path]

    print("END async_gather_patch")
    return True


async def npu_worker(input_Q, output_Q, submitter):
    print("START NPU_WORKER")
    await asyncio.sleep(1)

    while True:
        input_data = input_Q.get()
        if input_data is None:
            await output_Q.put(None)
            break

        patch, y_idx, x_idx, data_path = input_data

        await submitter.submit(patch, context=(y_idx, x_idx, data_path))
        await output_Q.put(True)
    print("END NPU_WORKER")
    return True


async def async_main():
    random.seed(42)

    manager = mp.Manager()
    video_Q = manager.Queue(1024)
    patch_Q = manager.Queue(1024)
    output_Q = asyncio.Queue(1024)

    gray_Qs = dict()
    result_Qs = dict()
    for video_path in DATA_PATHS:
        gray_Qs[video_path] = manager.Queue(1024)
        result_Qs[video_path] = manager.Queue(1024)

    lock = manager.Lock()
    asyncio_lock = asyncio.Lock()

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(32) as p_executor:
        async with create_queue(
            onnx_path,
            worker_num=2,
            device="npu0pe0,npu0pe1",
        ) as (submitter, reciever):
            start = time.time()
            for video_path in DATA_PATHS:
                print(f"Read video {video_path}")
                p_executor.submit(video_handler, video_path, video_Q, gray_Qs)
            merger_l = [
                i
                for zip_l in zip(list(result_Qs.values()), list(gray_Qs.values()))
                for i in zip_l
            ]
            p_executor.submit(MERGER, "Pix2Pix", merger_l)

            postproc_task = asyncio.create_task(
                async_gather_patch(output_Q, result_Qs, reciever)
            )
            npu_task = asyncio.create_task(npu_worker(patch_Q, output_Q, submitter))
            preproc_task = asyncio.create_task(
                async_create_patch(p_executor, loop, video_Q, patch_Q)
            )

            await preproc_task
            await npu_task
            await postproc_task

    end = time.time()
    total_time = end - start


class NPU_runner:
    npu_task = None

    @staticmethod
    def kill_child_processes(sig=signal.SIGTERM):
        try:
            parent = psutil.Process(os.getpid())
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)

    @classmethod
    def startup(cls):
        if cls.npu_task:
            print("WARN: NPU runner started more than once.")
        cls.npu_task = asyncio.create_task(async_main())

    @classmethod
    def shutdown(cls):
        if not cls.npu_task:
            print("WARN: NPU runner not started yet. Ignore stopping NPU runner")
            return
        cls.npu_task.cancel()
        cls.kill_child_processes(signal.SIGKILL)


def video_handler(video_path, video_Q, gray_Qs):
    print(f"start processing {video_path}")

    while True:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        img_idx = 0
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            img_path = os.path.join(video_path, "%010d.bmp" % (img_idx))
            video_Q.put((frame, img_path))
            gray_Qs[video_path].put(frame)
            img_idx += 1

        if cap.isOpened():
            cap.release()
    video_Q.put(None)
    return


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


@app.get("/chart_data")
async def get_data():
    def generate_data():
        power, util, temp, se, devices = warboy_device()
        return jsonable_encoder(
            {"power": power, "util": util, "temp": temp, "time": se, "devices": devices}
        )

    d = generate_data()
    return JSONResponse(content=d)


@app.on_event("startup")
async def init_npu_runner():
    NPU_runner.startup()


@app.on_event("shutdown")
async def destroy_npu_runner():
    NPU_runner.shutdown()


if __name__ == "__main__":
    uvicorn.run(app="colorization_web:app", host="0.0.0.0", port=20005, reload=False)
