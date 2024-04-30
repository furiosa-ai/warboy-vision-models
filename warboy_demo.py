import multiprocessing as mp
import sys,os
import subprocess
import threading
import time

import cv2
import psutil
import typer
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from utils.handler import InputHandler, OutputHandler
from utils.result_img_process import ImageMerger
from utils.mp_queue import MpQueue, QueueStopEle
from utils.parse_params import get_demo_params_from_cfg, get_output_paths
from utils.postprocess import getPostProcesser
from utils.preprocess import YOLOPreProcessor
from utils.warboy_runner import WarboyRunner

app = FastAPI()


class AppRunner:
    def __init__(self, param, result_queues):
        self.app_type = param["app"]
        self.video_paths = param["video_paths"]
        self.runtime_params = param["runtime_params"]

        #if os.path.exists(param["output_path"]):
        #    subprocess.run(["rm", "-rf", param["output_path"]])
        #os.makedirs(param["output_path"])

        self.input_queues = [MpQueue(75) for _ in range(len(self.video_paths))]
        self.frame_queues = [MpQueue(75) for _ in range(len(self.video_paths))]
        self.output_queues = [MpQueue(50) for _ in range(len(self.video_paths))]
        self.result_queues = result_queues
        self.furiosa_runtime = WarboyRunner(
            param["model_path"], param["worker_num"], param["warboy_device"]
        )
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = getPostProcesser(
            self.app_type,
            param["model_name"],
            self.runtime_params,
            param["class_names"],
        )

        self.input_handler = InputHandler(
            self.video_paths,
            param["output_path"],
            self.input_queues,
            self.frame_queues,
            self.preprocessor,
            param["input_shape"],
        )
        self.output_handler = OutputHandler(
            self.video_paths,
            param["output_path"],
            self.output_queues,
            self.frame_queues,
            self.result_queues,
            self.postprocessor,
            draw_fps=True,
        )

    def __call__(self):
        warboy_runtime_process = mp.Process(
            target=self.furiosa_runtime, args=(self.input_queues, self.output_queues)
        )
        self.input_handler.start()
        warboy_runtime_process.start()
        self.output_handler.start()

        self.input_handler.join()
        for input_q in self.input_queues:
            input_q.put(QueueStopEle)
        
        for frame_queue in self.frame_queues:
            frame_queue.put(QueueStopEle)
            
        warboy_runtime_process.join()
        for output_queue in self.output_queues:
            output_queue.put(QueueStopEle)
        self.output_handler.join()

        for result_queue in self.result_queues:
            result_queue.put(QueueStopEle)

        print(f"Application -> {self.app_type} End!!")


class DemoApplication:
    def __init__(self, cfg):
        self.cfg = cfg
        self.demo_params = get_demo_params_from_cfg(cfg)
        self.result_queues = []
        self.app_runners = []
        self.merger = ImageMerger()

        for param in self.demo_params:
            app_result_queues = [MpQueue(50) for _ in range(len(param["video_paths"]))]
            self.app_runners.append(
                AppRunner(param, app_result_queues)
            )
            self.result_queues += app_result_queues

        self.app_threads = [
            threading.Thread(target=app_runner, args=())
            for app_runner in self.app_runners
        ]

    def run(
        self,
    ):

        merge_proc = mp.Process(target=self.merger, args = (self.result_queues, (720,1280), ))
        for app_thread in self.app_threads:
            app_thread.start()
        merge_proc.start()
        for app_thread in self.app_threads:
            app_thread.join()
        merge_proc.join()
        return


def run_demo_thread(demo_application):
    t = threading.Thread(target = demo_application.run)
    t.daemon = True
    t.start()
    return t.native_id

def getByteFrame():
    cnt = 0
    while True:
        img_path = os.path.join("result", "%010d.bmp"%cnt)
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


@app.get("/")
def stream():
    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

def run_web_server():
    proc = mp.Process(
        target = uvicorn.run,
        args = ("warboy_demo:app",),
        kwargs = {
            "host": "0.0.0.0",
            "port": 20001,
        },
    )
    proc.start()
    return proc

def shutdown_web_server(proc):
    pid = proc.pid
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    proc.terminate()
    for _ in range(5):
        if proc.is_alive():
            time.sleep(1)
            print("Alive Web Server..")
    
    if proc.is_alive():
        subprocess.run(["kill","-9",str(pid)])
    return

if __name__ == "__main__":
    assert len(sys.argv) == 2, "len(argument) must be 3"
    if os.path.exists("result"):
        subprocess.run(["rm","-rf", "result"])
    os.makedirs("result")

    demo_app = DemoApplication(sys.argv[1])
    run_demo_thread(demo_app)
    proc = run_web_server()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_web_server(proc)
        pass
   
    print("EXIT!!!!")