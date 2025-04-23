import time
from typing import Callable

from src.warboy.utils.queue import PipeLineQueue, QueueClosedError, StopSig


class ImageEncoder:
    def __init__(
        self,
        frame_mux: PipeLineQueue,
        output_mux: PipeLineQueue,
        result_mux: PipeLineQueue,
        postprocess_function: Callable,
    ):
        self.frame_mux = frame_mux
        self.output_mux = output_mux
        self.result_mux = result_mux
        self.postprocessor = postprocess_function
        pass

    def run(self):
        FPS = 0.0
        curr_idx = 0
        num_comp = 0
        start_time = time.time()
        while True:
            try:
                frame, context, img_idx = self.frame_mux.get()
                output = self.output_mux.get()
                annotated_img = self.postprocessor(output, context, frame)
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    FPS = (curr_idx - num_comp) / elapsed_time
                    start_time = time.time()
                    num_comp = curr_idx

                if not self.result_mux is None:
                    self.result_mux.put((annotated_img, FPS, img_idx))
                    curr_idx += 1
            except QueueClosedError:
                if not self.result_mux is None:
                    self.result_mux.put(StopSig)
                break
            except Exception as e:
                print(f"Error ImageEncoder: {e}")
                break


class PredictionEncoder:
    def __init__(
        self,
        frame_mux: PipeLineQueue,
        output_mux: PipeLineQueue,
        result_mux: PipeLineQueue,
        postprocess_function: Callable,
    ):
        self.frame_mux = frame_mux
        self.output_mux = output_mux
        self.result_mux = result_mux
        self.postprocessor = postprocess_function

    def run(self):
        while True:
            try:
                frame, context, img_idx = self.frame_mux.get()
                output = self.output_mux.get()
                preds = self.postprocessor(output, context, frame.shape[:2])
                if not self.result_mux is None:
                    self.result_mux.put((preds, 0.0, img_idx))
            except QueueClosedError:
                if not self.result_mux is None:
                    self.result_mux.put(StopSig)
                break
            except Exception as e:
                print(f"Error PredictionEncoder: {e}")
                break
