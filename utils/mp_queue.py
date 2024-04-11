from multiprocessing import Lock as _MPLock
from multiprocessing import Queue as _MPQueue


class QueueStopEle:
    pass


class QueueClosedError(Exception):
    pass


class MpQueue:
    def __init__(self, max_size: int = 0) -> None:
        self.qu = _MPQueue(maxsize=max_size)
        self.get_lk = _MPLock()

    def put(self, item, block=True, timeout=None) -> None:
        # assert item is not _QueueStopEle
        return self._put(item, block, timeout)

    def _put(self, item, block=True, timeout=None) -> None:
        return self.qu.put(item, block, timeout)

    def get(self, block=True, timeout=None):
        with self.get_lk:
            if self.qu is not None:
                item = self.qu.get(block, timeout)
            else:
                item = QueueStopEle

            if item is QueueStopEle:
                if self.qu is not None:
                    self.qu.put(item)  # propagate if multiple threads are using same qu
                    # self.qu = None
                raise QueueClosedError

        return item

    def close(self, num_workers=1):
        for _ in range(num_workers):
            self._put(QueueStopEle)
        self.qu = None
