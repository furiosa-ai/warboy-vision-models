import asyncio
import time
from collections import defaultdict
from typing import List

from furiosa.runtime import create_queue
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig

from warboy.utils.queue import PipeLineQueue, QueueClosedError, StopSig


class WarboyApplication:
    """
    An inference engine using FuriosaAI Runtime, based on Queue

    Args:
        model(str): a path to quantized onnx file
        worker_num(int): the number of npu workers
        device(str): a set of NPU devices by a textual string (e.g. "warboy(2)*1", "npu0pe0", etc)
        stream_mux_list(list):
        output_mux_list(list):
    """

    def __init__(
        self,
        model: str,
        worker_num: str,
        device: str,
        stream_mux_list: List[PipeLineQueue],
        output_mux_list: List[PipeLineQueue],
    ):
        self.config = {"model": model, "worker_num": worker_num, "npu_device": device}
        self.model = FuriosaRTModel(
            FuriosaRTModelConfig(name="YOLO", batch_size=1, **self.config)
        )
        self.stream_mux_list = stream_mux_list
        self.output_mux_list = output_mux_list
        print("WarboyApplication - init")

    def run(self):
        asyncio.run(self.task())

    async def task(self):
        await self.load()
        await asyncio.gather(
            *(
                self.inference(video_channel, stream_mux, output_mux)
                for video_channel, (stream_mux, output_mux) in enumerate(
                    zip(self.stream_mux_list, self.output_mux_list)
                )
            )
        )
        return

    async def inference(
        self, video_channel: int, stream_mux: PipeLineQueue, output_mux: PipeLineQueue
    ):

        while True:
            t1 = time.time()
            try:
                input_, _ = stream_mux.get()
            except QueueClosedError:
                # print(f"Video-Channel - {video_channel} End!")
                break
            output = await self.model.predict(input_)
            output_mux.put(output)

        output_mux.put(StopSig)
        return

    async def load(self):
        await self.model.load()


class WarboyQueueRuntime:
    def __init__(
        self,
        model: str,
        worker_num: int,
        device: str,
        stream_mux_list: List[PipeLineQueue],
        output_mux_list: List[PipeLineQueue],
    ):
        self.config = {"model": model, "worker_num": worker_num, "device": device}
        self.submitter = None
        self.receiver = None
        self.stream_mux_list = stream_mux_list
        self.output_mux_list = output_mux_list
        # TEST
        self.pending_tasks = 0
        self.stop_count = 0
        self.total_submitters = len(self.stream_mux_list)

        self.pending_lock = None
        self.done_event = None

        print("WarboyQueueRuntime - init")

    def run(self):
        asyncio.run(self.run_())

    async def run_(self):
        self.submitter, self.receiver = await create_queue(**self.config)
        # TEST
        self.pending_lock = asyncio.Lock()
        self.done_event = asyncio.Event()

        task = [self.recv_with()] + [
            self.submit_with(video_channel)
            for video_channel in range(len(self.stream_mux_list))
        ]
        await asyncio.gather(*task)

    async def submit_with(self, video_channel: int):

        while True:
            try:
                input_, img_idx = self.stream_mux_list[video_channel].get()

                async with self.pending_lock:
                    self.pending_tasks += 1

                await self.submitter.submit(input_, context=(video_channel, img_idx))
            except QueueClosedError:
                async with self.pending_lock:
                    self.stop_count += 1
                    if (
                        self.stop_count == self.total_submitters
                        and self.pending_tasks == 0
                    ):
                        self.done_event.set()
                        break
                print(f"Channel - {video_channel} End!")
                break
            except Exception as e:
                print("Line 43 - submit:", e)
                break
        return

    async def recv_with(self):

        # buffer: channel → {img_idx → output}
        buffer = defaultdict(dict)
        # expected next index: channel → int
        expected_idx = defaultdict(lambda: 0)

        while True:
            try:
                t1 = time.time()
                (video_channel, img_idx), output = await self.receiver.recv()

                buffer[video_channel][img_idx] = output
                while expected_idx[video_channel] in buffer[video_channel]:
                    output = buffer[video_channel].pop(expected_idx[video_channel])
                    self.output_mux_list[video_channel].put(output)
                    expected_idx[video_channel] += 1

                async with self.pending_lock:
                    self.pending_tasks -= 1
                    if (
                        self.stop_count == self.total_submitters
                        and self.pending_tasks == 0
                    ):
                        self.done_event.set()
                        break
            except asyncio.TimeoutError:
                print("TimeOut Receiver")
                break
            except Exception as e:
                print("Line 61 - recv:", e)
                break

        for output_mux in self.output_mux_list:
            output_mux.put(StopSig)
        return
