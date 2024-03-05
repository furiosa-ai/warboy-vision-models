import datetime
import math
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from typing import List, Tuple

import cv2
from furiosa.device.sync import list_devices
import numpy as np


class ResultViewer:
    def __init__(
        self,
        output_paths: List[str],
        full_grid_shape: Tuple[int, int] = (1080, 1920),
        viewer: str = 'fastAPI',
    ):
        self.full_grid_shape = full_grid_shape
        self.output_paths = output_paths
        self.num_grid = math.ceil(math.sqrt(len(output_paths)))
        self.grid_shape = (
            int((full_grid_shape[0] - 5) / self.num_grid) - 5,
            int((full_grid_shape[1] - 5) / self.num_grid) - 5,
        )
        self.viewer = viewer

    def draw_img_to_grid_video(self):
        num_channel = len(self.output_paths)
        end_channel = 0

        img_idx = 0
        states = [True for _ in range(num_channel)]
        result_path = "result"

        if self.viewer == "file":
            if os.path.exists(result_path):
                subprocess.run(["rm", "-rf", result_path])
            os.makedirs(result_path)

        while True:
            if img_idx == 100:
                break

            if end_channel == num_channel:
                break

            channel_idx = 0
            grid_imgs = []

            while True:
                if channel_idx == num_channel:
                    break

                output_img_path = os.path.join(
                    self.output_paths[channel_idx], "%010d.bmp" % img_idx
                )
                last_file_path = os.path.join(self.output_paths[channel_idx], "%010d.csv" % img_idx)

                if os.path.exists(output_img_path) and states[channel_idx]:
                    channel_img = cv2.imread(output_img_path)
                    grid_img = cv2.resize(
                        channel_img,
                        (self.grid_shape[1], self.grid_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    grid_imgs.append(grid_img)
                    channel_idx += 1
                    os.remove(output_img_path)
                elif os.path.exists(last_file_path) or not states[channel_idx]:
                    if states[channel_idx]:
                        end_channel += 1
                    states[channel_idx] = False
                    grid_imgs.append(None)
                    channel_idx += 1
            out_img = self.make_img_grid(grid_imgs)

            if self.viewer == "file":
                result_img_path = os.path.join(result_path, "%010d.bmp" % img_idx)
                cv2.imwrite(result_img_path, out_img)
            elif self.viewer == "open-cv":
                cv2.imshow(self.viewer, out_img)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
            elif self.viewer == "fastAPI":
                out_img = cv2.imencode('.jpg', out_img)
                out_frame = out_img.tobytes()
                yield_frame(out_frame)
            else:
                pass

            img_idx += 1

    def make_img_grid(self, grid_imgs):
        full_grid = np.zeros((self.full_grid_shape[0], self.full_grid_shape[1], 3), np.uint8)
        for i, grid in enumerate(grid_imgs):
            if grid is None:
                continue

            r = i // self.num_grid
            c = i % self.num_grid

            x1 = r * self.grid_shape[0] + (r + 1) * 5
            x2 = (r + 1) * self.grid_shape[0] + (r + 1) * 5
            y1 = c * self.grid_shape[1] + (c + 1) * 5
            y2 = (c + 1) * self.grid_shape[1] + (c + 1) * 5
            full_grid[x1:x2, y1:y2] = grid

        return full_grid


class WarboyViewer:
    def __init__(self):
        self.state = True
        self.devices = list_devices()
        self.proc = threading.Thread(target=self.view_npu_utilization)

    def start(self):
        self.proc.start()

    def join(self):
        self.proc.join()

    def view_npu_utilization(self):
        last_pc = {}

        ts = time.time()
        while self.state:
            os.system("clear")
            timer = datetime.datetime.now()
            device_states = []
            for device in self.devices:
                device_name = str(device)
                per_counters = device.performance_counters()

                state = ""
                for per_counter in per_counters:
                    pe_name = str(per_counter[0])
                    cur_pc = per_counter[1]

                    if pe_name in last_pc:
                        warboy_util, comp_ratio, io_ratio = self.calculate_result(
                            cur_pc, last_pc[pe_name]
                        )
                        device_ = "{0:<9}".format(pe_name)
                        graph, util, comp, io = self.state_to_str(warboy_util, comp_ratio, io_ratio)
                        state += "{}:{} {} {} {}\n".format(device_, graph, util, comp, io)
                    last_pc[pe_name] = cur_pc

                if len(state) == 0:
                    device_ = "{0:<9}".format(device_name)
                    graph, util, comp, io = self.state_to_str(0.0, 0.0, 0.0)
                    state += "{}:{} {} {} {}\n".format(device_, graph, util, comp, io)
                device_states.append(state)

            print(f"Datetime: {timer}, Run-Time: {time.time()-ts}s")
            print("-" * 70)
            print("{0:9}  {1:<34} Util(%) Comp(%) I/O(%)".format("Device", "Status- Comp:▓  I/O:▒"))
            for device_state in device_states:
                print(device_state, end="\r")
            print("-" * 70)
            time.sleep(0.3)

    def calculate_result(self, cur_pc, last_pc):
        result = cur_pc.calculate_utilization(last_pc)

        warboy_util = result.npu_utilization()
        comp_ratio = result.computation_ratio()
        io_ratio = result.io_ratio()

        return warboy_util, comp_ratio, io_ratio

    def state_to_str(self, warboy_util, comp_ratio, io_ratio):
        comp_graph = (int(round(comp_ratio * warboy_util * 100 / 3))) * "▓"
        io_graph = (int(round(io_ratio * warboy_util * 100 / 3))) * "▒"
        graph = "[{0: <33}]".format(comp_graph + io_graph)
        util = "% 7.2f" % (warboy_util * 100)

        comp = "{0:>7}".format("-")
        io = "{0:>6}".format("-")
        if warboy_util != 0.0:
            comp = "% 7.2f" % (comp_ratio * 100)
            io = "% 6.2f" % (io_ratio * 100)

        return graph, util, comp, io


def yield_frame(frame):
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(out_frame) + b'\r\n')
