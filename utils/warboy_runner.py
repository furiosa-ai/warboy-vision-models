import asyncio
import os
import queue
import subprocess
import threading
import time
from typing import List

import numpy as np
import cv2
from furiosa import runtime
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from utils.mp_queue import MpQueue, QueueClosedError


class WarboyRunner:
    """ """

    def __init__(
        self, model_path: str, worker_num: int = 8, device: str = "warboy(2)*1"
    ):
        self.model_path = model_path
        self.worker_num = worker_num
        self.device = device

    def __call__(self, input_queue: MpQueue, output_queues: List[MpQueue]):
        asyncio.run(self.runner(input_queue, output_queues))

    async def runner(self, input_queue: MpQueue, output_queues: List[MpQueue]):
        async with runtime.create_queue(
            model=self.model_path, worker_num=self.worker_num, device=self.device
        ) as (submitter, receiver):
            submit_task = asyncio.create_task(self.submit_with(submitter, input_queue))
            recv_task = asyncio.create_task(self.recv_with(receiver, output_queues))
            await submit_task
            await recv_task

    async def submit_with(self, submitter, input_queue):
        while True:
            try:
                input_, contexts, img_idx, video_idx = input_queue.get()
            except QueueClosedError:
                break
            await submitter.submit(input_, context=(contexts, img_idx, video_idx))

    async def recv_with(self, receiver, output_queues):
        while True:

            async def recv():
                context, outputs = await receiver.recv()
                return context, outputs

            try:
                recv_task = asyncio.create_task(recv())
                (contexts, img_idx, video_idx), outputs = await asyncio.wait_for(
                    recv_task, timeout=1
                )
            except asyncio.TimeoutError:
                break

            try:
                output_queues[video_idx].put((outputs, contexts, img_idx))
            except queue.Full:
                time.sleep(0.001)
                output_queues[video_idx].put((outputs, contexts, img_idx))

class WarboyServer:
    def __init__(self, model_path: str, worker_num: int = 8, device: str = "warboy(2)*1", preprocessor = None, postprocessor=None, input_shape=None, output_path=None, video_names = None):
        self.model = FuriosaRTModel(
            FuriosaRTModelConfig(
                name = "model",
                model = model_path,
                worker_num = worker_num,
                npu_device = device,
            )
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.video_names = video_names
        self.input_shape = input_shape
        self.output_path = output_path
    
    async def load(self):
        await self.model.load()

    async def process(self, video_name, img_idx, worker_id):
        img_path = os.path.join(self.output_path, "_input", video_name, "%010d.bmp" % img_idx)
        end_file = os.path.join(self.output_path, "_input", video_name, "%010d.csv" % img_idx)
        if os.path.exists(end_file):
            return 1

        if not os.path.exists(img_path):
            return -1

        t1 = time.time()
        org_img = cv2.imread(img_path)
        input_, contexts = self.preprocessor(org_img, new_shape = self.input_shape)
        out = await self.model.predict(input_)
        out_img = self.postprocessor[worker_id](out, contexts, org_img)
        t2 = time.time()

        FPS = 1 / (t2-t1)
        output_img = put_fps_to_img(out_img , FPS)
        dummy_path = os.path.join(self.output_path,  "_output", video_name, ".%010d.bmp" % img_idx)
        out_img_path = os.path.join(self.output_path, "_output", video_name, "%010d.bmp" % img_idx)
        cv2.imwrite(dummy_path, output_img)
        os.rename(dummy_path, out_img_path)
        return 0

    async def run(self):
        await asyncio.gather(*(self.task(self.video_names[worker_id], worker_id) for worker_id in range(len(self.video_names))))


    async def task(self, video_name, worker_id):
        output_path = os.path.join(self.output_path, "_output", video_name)
        os.makedirs(output_path)
        img_idx = 0
        worker_state = 0
        while True:
            res = await self.process(video_name, img_idx, worker_id)
            if res == 1:
                img_idx = 0
                continue
            elif res < 0 :
                continue
            img_idx += 1

        return

def put_fps_to_img(output_img: np.ndarray, FPS: float):
    h, w, _ = output_img.shape
    org = (int(0.05 * h), int(0.05 * w))
    scale = int(org[1] * 0.07)

    output_img = cv2.putText(
        output_img,
        f"FPS: {FPS:.2f}",
        (int(0.05 * h), int(0.05 * w)),
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 0, 0),
        scale,
        cv2.LINE_AA,
    )
    return output_img

