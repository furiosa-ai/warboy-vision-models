import asyncio
import queue
from typing import List, Dict, Any

from furiosa.runtime import create_queue
from furiosa.runtime import create_runner
from warboy.runtime.mp_queue import MpQueue, QueueStopEle, QueueClosedError


class WarboyRuntimeQueue:
    """
    An inference engine using FuriosaAI Runtime, which is based on Queues.

    Args:
        model_path(str) :
        worker_num(int) : The number of npu workers
        device(str) : a set of NPU devices by a textual string.
        midx(int) : 
    Methods:

    """

    def __init__(
        self,
        model_path: str,
        worker_num: int = 8,
        device: str = "warboy(2)*1",
        midx: int = 0,
    ) -> None:
        self.model_path = model_path
        self.worker_num = worker_num
        self.device = device
        self.midx = midx

    def __call__(self, input_queue: MpQueue, output_queues: List[MpQueue]):
        asyncio.run(self.run(input_queue, output_queues))

    async def run(
        self, input_queues: List[List[MpQueue]], output_queues: List[List[MpQueue]]
    ) -> None:
        async with create_queue(
            model=self.model_path, worker_num=self.worker_num, device=self.device, compiler_config={"lower_tabulated_dequantize": True}
        ) as (submitter, receiver):
            submit_task = asyncio.create_task(self.submit_with(submitter, input_queues))
            recv_task = asyncio.create_task(self.recv_with(receiver, output_queues))
            await submit_task
            await recv_task

    async def submit_with(self, submitter, input_queues: List[List[MpQueue]]) -> None:
        vidx = 0
        num_stop_queues = 0
        num_videos = len(input_queues)
        queue_states = [True for _ in range(num_videos)]
        while num_stop_queues < num_videos:
            vidx = vidx % num_videos
            state = queue_states[vidx]

            while state:
                try:
                    input_ = input_queues[vidx][self.midx].get(False)
                    await submitter.submit(input_, context=(vidx))
                    break
                except queue.Empty:
                    await asyncio.sleep(1e-6)
                except QueueClosedError:
                    queue_states[vidx] = False
                    num_stop_queues += 1
            vidx += 1
        return

    async def recv_with(self, receiver, output_queues: List[List[MpQueue]]) -> None:

        while True:
            try:
                video_idx, outputs = await receiver.recv()
                output_queues[video_idx][self.midx].put(outputs)
            except asyncio.TimeoutError:
                break

        for output_queue in output_queues:
            for oq in output_queue:
                oq.put(QueueStopEle)
        return


class WarboyRuntimeRunner:
    def __init__(
        self,
        model_path: str,
        worker_num: int = 8,
        device: str = "warboy(2)*1",
        midx: int = 0,
    ) -> None:
        self.model_path = model_path
        self.worker_num = worker_num
        self.device = device
        self.midx = midx
        self.num_task = None

    def __call__(self, input_queue: MpQueue, output_queues: List[MpQueue]):
        asyncio.run(self.run(input_queue, output_queues))

    async def run(
        self, input_queues: List[List[MpQueue]], output_queues: List[List[MpQueue]]
    ) -> None:
        num_task = len(input_queues)
        self.num_task = len(input_queues)
        async with create_runner(
            self.model_path, worker_num=self.worker_num, device=self.device, compiler_config={"lower_tabulated_dequantize": True}
        ) as runner:
            await asyncio.gather(
                *(
                    self.task(
                        runner,
                        input_queues[idx][self.midx],
                        output_queues[idx][self.midx],
                    )
                    for idx in range(num_task)
                )
            )
        return

    async def task(self, runner, input_queue: MpQueue, output_queue: MpQueue):
        while True:
            try:
                input_ = input_queue.get(False)
                preds = await runner.run([input_])
                output_queue.put((preds))
            except queue.Empty:
                await asyncio.sleep(0)
            except QueueClosedError:
                break

        output_queue.put(QueueStopEle)
        return
