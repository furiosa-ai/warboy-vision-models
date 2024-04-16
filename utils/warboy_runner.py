import asyncio
import os
import queue
import subprocess
import threading
import time
from typing import List

from furiosa import runtime

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
