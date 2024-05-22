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
from contextlib import asynccontextmanager
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

# >>> EDITABLE CONFIGS >>>
# path config
ONNX_PATH = "colorization_after_1_best_i8_origin.onnx"
DATA_PATHS = glob.glob("video/*.wmv")

# data config
IN_PATCH_SIZE = 256
## number of patches
NUM_WIDTH = 1
NUM_HEIGHT = 1
## pad size to cut off from raw output
PAD = 32
## size of overlap from cut offed output
OVERLAP = 0

# worker config
NUM_INPUT_WORKER = 8
# <<< EDITABLE CONFIGS <<<


# >>> DO NOT EDIT >>>
_SINGLE_PAD = PAD // 2

_IN_STRIDE = IN_PATCH_SIZE - OVERLAP - PAD

_OUT_PATCH_SIZE = IN_PATCH_SIZE - PAD
_OUT_STRIDE = _OUT_PATCH_SIZE - OVERLAP

_INPUT_HEIGHT = NUM_HEIGHT * _IN_STRIDE + OVERLAP + PAD
_INPUT_WIDTH = NUM_WIDTH * _IN_STRIDE + OVERLAP + PAD
_OUTPUT_HEIGHT = NUM_HEIGHT * _OUT_STRIDE + OVERLAP
_OUTPUT_WIDTH = NUM_WIDTH * _OUT_STRIDE + OVERLAP
# <<< DO NOT EDIT <<<

_warboy_device = WarboyDevice()


class Colorizer:
    def __init__(self):
        self.manager = mp.Manager()
        self.video_Q = self.manager.Queue(1024)
        self.patch_Q = self.manager.Queue(1024)
        self.output_Q = asyncio.Queue(1024)

        self.gray_Qs = dict()
        self.result_Qs = dict()
        for video_path in DATA_PATHS:
            self.gray_Qs[video_path] = self.manager.Queue(1024)
            self.result_Qs[video_path] = self.manager.Queue(1024)

    @staticmethod
    def video_split_worker(video_path, video_Q, gray_Qs):
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

    def launch_video_workers(self, p_executor):
        merger = ImageMerger()

        # Preprocessing
        for video_path in DATA_PATHS:
            print(f"Read video {video_path}")
            p_executor.submit(
                Colorizer.video_split_worker, video_path, self.video_Q, self.gray_Qs
            )
        merger_l = [
            i
            for zip_l in zip(list(self.result_Qs.values()), list(self.gray_Qs.values()))
            for i in zip_l
        ]
        # Postprocessing
        p_executor.submit(merger, "Pix2Pix", merger_l)

    @staticmethod
    def patch_creator(video_Q, patch_Q):
        def create_patch():
            out = video_Q.get()
            if not out:
                return out
            img, data_path = out
            img = cv2.resize(
                img, (_INPUT_WIDTH, _INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
            h, w, _ = img.shape
            for h_idx in range(NUM_HEIGHT):
                for w_idx in range(NUM_WIDTH):
                    hh = h_idx * _IN_STRIDE
                    ww = w_idx * _IN_STRIDE

                    patch = img[hh : hh + IN_PATCH_SIZE, ww : ww + IN_PATCH_SIZE]
                    patch = patch.transpose((2, 0, 1))[::-1].copy()
                    patch = np.ascontiguousarray(
                        np.expand_dims(patch, 0), dtype=np.uint8
                    )
                    patch_Q.put((patch, h_idx, w_idx, data_path))
            return True

        while create_patch():
            pass

    async def create_patch_worker(self, executor, loop):
        print("START create_patch_worker")

        exec_l = list()
        for _ in range(NUM_INPUT_WORKER):
            exec_l.append(
                loop.run_in_executor(
                    executor, Colorizer.patch_creator, self.video_Q, self.patch_Q
                )
            )

        for t in exec_l:
            await t

        print("END preproc_task")
        self.patch_Q.put(None)
        return True

    async def gather_patch_worker(self, reciever):
        print("START gather_patch_worker")
        await asyncio.sleep(1)

        def gather_patch(img_mat):
            num_h = len(img_mat)
            num_w = len(img_mat[0])

            full_img = np.zeros(
                (
                    _OUTPUT_HEIGHT,
                    _OUTPUT_WIDTH,
                    3,
                )
            )

            for h_idx in range(num_h):
                for w_idx in range(num_w):
                    out_patch = img_mat[h_idx][w_idx][
                        _SINGLE_PAD : _SINGLE_PAD + _OUT_PATCH_SIZE,
                        _SINGLE_PAD : _SINGLE_PAD + _OUT_PATCH_SIZE,
                    ]
                    hh = h_idx * _OUT_STRIDE
                    ww = w_idx * _OUT_STRIDE
                    full_img[hh : hh + _OUT_PATCH_SIZE, ww : ww + _OUT_PATCH_SIZE] = (
                        np.add(
                            out_patch,
                            full_img[
                                hh : hh + _OUT_PATCH_SIZE, ww : ww + _OUT_PATCH_SIZE
                            ],
                        )
                    )

                    if h_idx == 0 and w_idx == 0:
                        continue
                    if h_idx > 0:
                        full_img[hh : hh + OVERLAP, ww : ww + _OUT_PATCH_SIZE] /= 2
                    if w_idx > 0:
                        full_img[hh : hh + _OUT_PATCH_SIZE, ww : ww + OVERLAP] /= 2
                    if h_idx > 0 and w_idx > 0:
                        full_img[hh : hh + OVERLAP, ww : ww + OVERLAP] *= 2

            return full_img

        curr_idx = -1
        data_path = -1
        img_mat = dict()
        img_mat_count = dict()

        while True:
            if await self.output_Q.get() is None:
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
                self.result_Qs[os.path.dirname(data_path)].put(
                    gather_patch(img_mat[data_path])
                )
                del img_mat[data_path]
                del img_mat_count[data_path]

        print("END gather_patch_worker")
        return True

    async def npu_worker(self, submitter):
        print("START NPU_WORKER")
        await asyncio.sleep(1)

        while True:
            input_data = self.patch_Q.get()
            if input_data is None:
                await self.output_Q.put(None)
                break

            patch, y_idx, x_idx, data_path = input_data

            await submitter.submit(patch, context=(y_idx, x_idx, data_path))
            await self.output_Q.put(True)
        print("END NPU_WORKER")
        return True

    async def run(self, p_executor):
        loop = asyncio.get_running_loop()

        async with create_queue(
            ONNX_PATH,
            worker_num=2,
            device="npu0pe0,npu0pe1",
        ) as (submitter, reciever):
            start = time.time()

            self.launch_video_workers(p_executor)

            postproc_task = asyncio.create_task(self.gather_patch_worker(reciever))
            npu_task = asyncio.create_task(self.npu_worker(submitter))
            preproc_task = asyncio.create_task(
                self.create_patch_worker(p_executor, loop)
            )

            await preproc_task
            await npu_task
            await postproc_task

            end = time.time()
            total_time = end - start


class NPU_runner:
    npu_task = None

    @staticmethod
    async def run():
        random.seed(42)
        colorizer = Colorizer()
        with ProcessPoolExecutor(32) as p_executor:
            await colorizer.run(p_executor)

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
        cls.npu_task = asyncio.create_task(NPU_runner.run())

    @classmethod
    def shutdown(cls):
        if not cls.npu_task:
            print("WARN: NPU runner not started yet. Ignore stopping NPU runner")
            return
        cls.kill_child_processes(signal.SIGKILL)
        # cls.npu_task.cancel()


# >>> Web Handler >>>


@asynccontextmanager
async def lifespan(app: FastAPI):
    NPU_runner.startup()
    yield
    NPU_runner.shutdown()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def stream():
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

    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/chart_data")
async def get_data():
    def generate_data():
        power, util, temp, se, devices = _warboy_device()
        return jsonable_encoder(
            {"power": power, "util": util, "temp": temp, "time": se, "devices": devices}
        )

    d = generate_data()
    return JSONResponse(content=d)


if __name__ == "__main__":
    try:
        shutil.rmtree(".tmp")
        Path(".tmp").mkdir(parents=True, exist_ok=True)
    except:
        pass
    uvicorn.run(app="colorization_web:app", host="0.0.0.0", port=20005, reload=False)
