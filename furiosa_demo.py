import multiprocessing as mp
import os
import subprocess
import time

import cv2
from demo import run_demo
import psutil
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


class DemoApplication:
    def __init__(self, cfg, viewer="fastAPI"):
        self.demo_process = mp.Process(target=run_demo, args=(cfg,))
        self.cfg = cfg
        self.viewer = viewer

    def run(
        self,
    ):
        self.demo_process.start()

        streaming_proc = subprocess.Popen(["python", "furiosa_stream.py", self.cfg, self.viewer])
        self.demo_process.join()
        time.sleep(100)
        self.shutdown_proc(streaming_proc)

    def shutdown_proc(self, process):
        pid = proc.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        proc.terminate()


@app.command()
def main(cfg, viewer):
    demo_app = DemoApplication(cfg, viewer)
    demo_app.run()


if __name__ == "__main__":
    app()
