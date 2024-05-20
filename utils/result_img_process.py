import math
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from utils.mp_queue import *
from furiosa.device.sync import list_devices

class ImageMerger:
    """ 
    """
    def __call__(
        self,
        model_names,
        result_queues,
        full_grid_shape: Tuple[int, int] = (1080, 1920),
    ):
        demo_model_names = []
        for model_name in model_names:
            for name in model_name:
                demo_model_names.append(name)
        num_channel = len(result_queues)
        ending_channel = 0

        num_grid, grid_shape = get_grid_info(num_channel, full_grid_shape)

        img_idx = [0 for _ in range(num_channel)]
        states = [True for _ in range(num_channel)]
        cnt = 0

        warboy_devices = list_devices()
        last_pc = {}

        while ending_channel < num_channel:
            c_idx = 0
            grid_imgs = []

            while c_idx < num_channel:
                try:
                    out_img = result_queues[c_idx].get()
                except QueueClosedError:
                    c_idx += 1
                    if states[c_idx]:
                        states[c_idx] = False
                        ending_channel += 1
                    grid_imgs.append(None)
                    continue
                
                grid_img = cv2.resize(
                    out_img,
                    (grid_shape[1], grid_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                grid_imgs.append(grid_img)
                c_idx += 1

            img = make_full_grid(grid_imgs, num_grid, grid_shape, full_grid_shape)
            output_path = ".tmp"
            cv2.imwrite(os.path.join(output_path, ".%010d.bmp" % cnt),img)
            os.rename(os.path.join(output_path, ".%010d.bmp" % cnt),os.path.join(output_path, "%010d.bmp" % cnt))
            cnt += 1
        return


def get_grid_info(num_channel: int, full_grid_shape: Tuple[int, int]):
    num_grid = math.ceil(math.sqrt(num_channel))
    grid_shape = (
        int((full_grid_shape[0] - 5) / num_grid) - 5,
        int((full_grid_shape[1] - 5) / num_grid) - 5,
    )
    return num_grid, grid_shape

def make_full_grid(
    grid_imgs: List[np.ndarray],
    num_grid: int,
    grid_shape: Tuple[int, int],
    full_grid_shape: Tuple[int, int],
) -> np.ndarray:
    pad = 10 # int(full_grid_shape[0] * 0.1)
    full_grid = np.zeros(
        (full_grid_shape[0] + pad, full_grid_shape[1] + pad, 3), np.uint8
    )

    for i, grid_img in enumerate(grid_imgs):
        if grid_img is None:
            continue

        r = i // num_grid
        c = i % num_grid

        x0 = r * grid_shape[0] + (r + 1) * 5
        x1 = (r + 1) * grid_shape[0] + (r + 1) * 5
        y0 = c * grid_shape[1] + (c + 1) * 5
        y1 = (c + 1) * grid_shape[1] + (c + 1) * 5
        full_grid[x0:x1, y0:y1] = grid_img

    return full_grid

def put_model_name_info(
    img: np.ndarray, model_names, img_shape: Tuple[int, int]
):
    initial_pos = (int(img_shape[1]*0.99), int(img_shape[0]*0.01) + 75)
    scale = max(int(0.003 * img_shape[0]), 1)

    for model_name in model_names:
        model_name = "| "+model_name
        t_size = cv2.getTextSize(model_name, 0, fontScale=scale, thickness=scale)[0]
        text_pos = (int(initial_pos[0]-t_size[0]//2-20), initial_pos[1]-25)
        img = cv2.putText(
            img,
            model_name,
            text_pos,
            cv2.FONT_HERSHEY_PLAIN,
            scale,
            (255, 255, 255),
            scale,
            cv2.LINE_AA,
        )
        initial_pos = (text_pos[0]-10, initial_pos[1])
    return img