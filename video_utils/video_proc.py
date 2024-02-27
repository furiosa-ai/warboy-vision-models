import multiprocessing as mp
import os
import queue
import subprocess

import cv2


class VideoProcessor:
    def __init__(self, video_paths, output_path, preproc, input_q, img_to_img=False):
        self.video_paths = video_paths
        self.output_path = output_path
        self.pre_proc = preproc
        self.procs = [
            mp.Process(target=self.video_to_frame, args=(video_path, idx, input_q, img_to_img))
            for idx, video_path in enumerate(video_paths)
        ]

    def start(self):
        for proc in self.procs:
            proc.start()

    def join(self):
        for proc in self.procs:
            proc.join()

    def video_to_frame(self, video_path, video_idx, input_q, img_to_img):
        video_name = (video_path.split('/')[-1]).split('.')[0]
        out_dir_path = os.path.join(self.output_path, "input", video_name)

        if os.path.exists(out_dir_path):
            subprocess.run(["rm", "-rf", out_dir_path])
        os.makedirs(out_dir_path)

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        img_idx = 0
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            input_, contexts = self.pre_proc(frame)
            try:
                input_q.put((input_, contexts, img_idx, video_idx))
            except queue.Full:
                time.sleep(0.01)
                input_q.put((input_, contexts, img_idx, video_idx))

            if not img_to_img:
                tmp_img_path = os.path.join(out_dir_path, "%.010d.bmp" % img_idx)
                out_img_path = os.path.join(out_dir_path, "%010d.bmp" % img_idx)

                cv2.imwrite(tmp_img_path, frame)
                os.rename(tmp_img_path, out_img_path)

            img_idx += 1

        if cap.isOpened():
            cap.release()

        return
