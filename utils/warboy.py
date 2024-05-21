import asyncio
import os
import queue
import subprocess
import threading
import time
from typing import List

from furiosa import runtime
from furiosa.device.sync import list_devices

from utils.mp_queue import MpQueue, QueueClosedError


class WarboyRunner:
    def __init__(
        self, model_path: str, worker_num: int = 8, device: str = "warboy(2)*1"
    ):
        self.model_path = model_path
        self.worker_num = worker_num
        self.device = device

    def __call__(
        self, input_queue: MpQueue, output_queues: List[MpQueue], model_idx: int
    ):
        asyncio.run(self.runner(input_queue, output_queues, model_idx))

    async def runner(
        self, input_queue: MpQueue, output_queues: List[MpQueue], model_idx: int
    ):
        async with runtime.create_queue(
            model=self.model_path, worker_num=self.worker_num, device=self.device
        ) as (submitter, receiver):
            len_videos = len(output_queues)
            submit_task = asyncio.create_task(
                self.submit_with(submitter, input_queue, len_videos, model_idx)
            )
            recv_task = asyncio.create_task(
                self.recv_with(receiver, output_queues, model_idx)
            )
            await submit_task
            await recv_task

    async def submit_with(self, submitter, input_queue, len_videos, model_idx):
        idx = 0
        while True:
            try:
                while True:
                    try:
                        input_, contexts, img_idx, video_idx = input_queue[
                            idx % len_videos
                        ][model_idx].get(False)
                        break
                    except queue.Empty:
                        await asyncio.sleep(0)

            except QueueClosedError:
                break

            await submitter.submit(input_, context=(contexts, img_idx, video_idx))
            idx += 1

    async def recv_with(self, receiver, output_queues, model_idx):
        queue_len = len(output_queues[0][0])
        while True:
            try:
                (contexts, img_idx, video_idx), outputs = await receiver.recv()
            except asyncio.TimeoutError:
                break
            output_queue = output_queues[video_idx][model_idx]
            output_queue[img_idx % queue_len].put((outputs, contexts, img_idx))


class WarboyDevice:
    def __init__(self):
        self.warboy_devices = list_devices()
        self.last_pc = {}
        self.idx = 0

    def __call__(self):
        power_info, util_info, temper_info, devices = get_warboy_info(
            self.warboy_devices, self.last_pc
        )
        self.idx += 1
        return power_info, util_info, temper_info, self.idx, devices


def get_warboy_info(devices, last_pc):
    powers = []
    utils = []
    tempers = []
    dd = []
    for device in devices:
        warboy_name = str(device)
        device_idx = warboy_name[3:]
        per_counters = device.performance_counters()
        if len(per_counters) != 0:
            fetcher = device.get_hwmon_fetcher()
            peak_device_temper = (
                int(str(fetcher.read_temperatures()[0]).split(" ")[-1]) // 1000
            )
            power_info = str(fetcher.read_powers_average()[0])
            p = int(float(power_info.split(" ")[-1]) / 1000000.0)
            powers.append(p)
            tempers.append(peak_device_temper)
            dd.append(device_idx)

        t_utils = 0.0
        for pc in per_counters:
            pe_name = str(pc[0])
            cur_pc = pc[1]

            if pe_name in last_pc:
                result = cur_pc.calculate_utilization(last_pc[pe_name])
                util = result.npu_utilization()
                if not ("0-1" in pe_name):
                    util /= 2.0
                t_utils += util

            last_pc[pe_name] = cur_pc

        if len(per_counters) != 0:
            t_utils = int(t_utils * 100.0)
            utils.append(t_utils)
    return powers, utils, tempers, dd
