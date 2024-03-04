import multiprocessing as mp
import os
import subprocess
import time
from typing import Sequence

import cv2
from pathlib import Path
from utils.mp_queue import *


class OutProcessor:
    def __init__(
        self,
        video_paths: Sequence[str],
        output_path: str,
        postproc,
        outputQs,
        draw_fps: bool = True,
        img_to_img: bool = False,
    ):
        self.output_path = output_path
        self.post_proc = postproc
        self.procs = [
            mp.Process(target=self.output_to_img, args=(video_path, outputQs[idx], draw_fps, img_to_img,))
            for idx, video_path in enumerate(video_paths)
        ]

    def start(self):
        for proc in self.procs:
            proc.start()

    def join(self):
        for proc in self.procs:
            proc.join()

    def output_to_img(
        self, video_path: str, outputQ: mp.Queue, draw_fps, img_to_img
    ):
        video_name = (video_path.split('/')[-1]).split('.')[0]

        in_img_dir_path = os.path.join(self.output_path, "input", video_name)
        out_dir_path = os.path.join(self.output_path, "output", video_name)

        if os.path.exists(out_dir_path):
            subprocess.run(["rm", "-rf", out_dir_path])
        os.makedirs(out_dir_path)

        img = None
        start_time = time.time()
        completed = 0
                
        while True:
            try:
                preds, contexts, img_idx = outputQ.get()
            except QueueClosedError:
                break

            if not img_to_img:
                in_img_path = os.path.join(in_img_dir_path, "%010d.bmp" % img_idx)
                img = cv2.imread(in_img_path)

            out_img = self.post_proc(outputs=preds, preproc_params=contexts, img=img)

            if draw_fps:
                FPS = completed / (time.time() - start_time)
                h, w, _ = out_img.shape
                org = (int(0.05 * h), int(0.05 * w))
                scale = int(org[1]*0.07)

                out_img = cv2.putText(
                    out_img,
                    f"FPS: {FPS:.2f}",
                    org,
                    cv2.FONT_HERSHEY_PLAIN,
                    scale,
                    (255,0,0),
                    scale,
                    cv2.LINE_AA
                )

            tmp_out_img_path = os.path.join(out_dir_path, ".%010d.bmp" % img_idx)
            out_img_path = os.path.join(out_dir_path, "%010d.bmp" % img_idx)

            cv2.imwrite(tmp_out_img_path, out_img)
            os.rename(tmp_out_img_path, out_img_path)

            completed += 1

        end_file = Path(os.path.join(out_dir_path, "%010d.csv" % completed))
        end_file.touch(exist_ok=True)
        if os.path.exists(in_img_dir_path):
            subprocess.run(["rm", "-rf", in_img_dir_path])

        return
