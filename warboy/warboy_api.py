import multiprocessing as mp
import threading
from typing import List, Dict, Any, Tuple

from warboy.runtime.mp_queue import MpQueue, QueueClosedError, QueueStopEle
from warboy.runtime.warboy_runtime import WarboyRuntimeQueue, WarboyRuntimeRunner
from warboy.utils.handler import Handler

QUEUE_SIZE = 100


class AppRunner:
    """

    """

    def __init__(self, param, result_queues: List[MpQueue]) -> None:
        num_task = len(param["task"])
        num_videos = len(param["videos_info"])

        self.input_queues = [
            [MpQueue(QUEUE_SIZE) for _ in range(num_task)] for _ in range(num_videos)
        ]
        self.frame_queues = [MpQueue(QUEUE_SIZE) for _ in range(num_videos)]
        self.output_queues = [
            [MpQueue(QUEUE_SIZE) for _ in range(num_task)] for _ in range(num_videos)
        ]

        # Warboy Runtime
        self.job_handler = Handler(
            self.input_queues, self.output_queues, result_queues, param
        )
        warboy_runtimes = self._get_warboy_runtime(param, num_task, "queue")

        self.warboy_runtime_procs = [
            mp.Process(
                target=warboy_runtime, args=(self.input_queues, self.output_queues)
            )
            for midx, warboy_runtime in enumerate(warboy_runtimes)
        ]

    def __call__(self):
        self.job_handler.start()
        for warboy_runtime_proc in self.warboy_runtime_procs:
            warboy_runtime_proc.start()
        self.job_handler.join()
        for warboy_runtime_proc in self.warboy_runtime_procs:
            warboy_runtime_proc.join()
        return

    def _get_warboy_runtime(self, param, num_task, runtime_type="runner"):
        if runtime_type == "queue":
            warboy_runtimes = [
                WarboyRuntimeQueue(
                    param["model_path"][i],
                    param["worker_num"],
                    param["warboy_device"][i],
                    i,
                )
                for i in range(num_task)
            ]
        elif runtime_type == "runner":
            warboy_runtimes = [
                WarboyRuntimeRunner(
                    param["model_path"][i],
                    param["worker_num"],
                    param["warboy_device"][i],
                )
                for i in range(num_task)
            ]
        else:
            raise ValueError(f"Warboy Runtime - (queue | runner)")

        return warboy_runtimes


class WARBOY_APP:
    """

    """

    def __init__(self, params):
        self.params = params

        self.result_queues = []
        self.app_runners = []

        manager = mp.Manager()
        for param in self.params:
            result_queues = [
                manager.Queue(QUEUE_SIZE) for _ in range(len(param["videos_info"]))
            ]
            self.app_runners.append(AppRunner(param, result_queues))
            self.result_queues += result_queues

        self.app_runner_threads = [
            threading.Thread(target=app_runner, args=())
            for app_runner in self.app_runners
        ]

    def get_result_queues(self):
        return self.result_queues

    def __call__(self):
        for app_runner_thread in self.app_runner_threads:
            app_runner_thread.start()

        for app_runner_thread in self.app_runner_threads:
            app_runner_thread.join()
