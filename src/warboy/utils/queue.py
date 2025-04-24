from multiprocessing import Queue
from typing import Any, Union


class StopSig:
    pass


class QueueClosedError(Exception):
    pass


class PipeLineQueue:
    def __init__(self, maxsize: int = 0):
        self.queue = Queue(maxsize=maxsize)

    def get(self, block: bool = True, timeout: Union[float, None] = None):
        item = self.queue.get(block=block, timeout=timeout)

        if item is StopSig:
            self.close()
            raise QueueClosedError

        return item

    def put(self, item: Any, block: bool = True, timeout: Union[float, None] = None):
        return self.queue.put(item)

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    def close(self):
        self.queue = None
