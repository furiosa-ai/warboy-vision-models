import math
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from furiosa.device.sync import list_devices


class ImageMerger:
    """ """

    def __call__(
        self,
        output_img_paths: List[str],
        full_grid_shape: Tuple[int, int] = (1080, 1920),
    ):
        num_channel = len(output_img_paths)
        ending_channel = 0

        num_grid, grid_shape = get_grid_info(num_channel, full_grid_shape)

        img_idx = 0
        states = [True for _ in range(num_channel)]

        warboy_devices = list_devices()
        last_pc = {}

        while ending_channel < num_channel:
            c_idx = 0
            grid_imgs = []

            while c_idx < num_channel:
                grid_img_path = os.path.join(
                    output_img_paths[c_idx], "%010d.bmp" % img_idx
                )
                if os.path.exists(grid_img_path):
                    img = cv2.imread(grid_img_path)
                    grid_img = cv2.resize(
                        img,
                        (grid_shape[1], grid_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    grid_imgs.append(grid_img)
                    c_idx += 1
                else:
                    ending_signal = os.path.join(
                        output_img_paths[c_idx], "%010d.csv" % img_idx
                    )
                    if os.path.exists(ending_signal) or not states[c_idx]:
                        if states[c_idx]:
                            ending_channel += 1
                        states[c_idx] = False
                        grid_imgs.append(None)
                        c_idx += 1

            img = make_full_grid(grid_imgs, num_grid, grid_shape, full_grid_shape)
            if img_idx % 5 == 0:
                power_info, util_info = get_warboy_info(warboy_devices, last_pc)
            img = put_warboy_info(img, power_info, util_info, full_grid_shape)
            yield (img)
            img_idx += 1
        return


def get_grid_info(num_channel: int, full_grid_shape: Tuple[int, int]):
    num_grid = math.ceil(math.sqrt(num_channel))
    grid_shape = (
        int((full_grid_shape[0] - 5) / num_grid) - 5,
        int((full_grid_shape[1] - 5) / num_grid) - 5,
    )
    return num_grid, grid_shape


def get_warboy_info(devices, last_pc):
    powers_str = ""
    utils_str = ""
    for device in devices:
        warboy_name = str(device)
        per_counters = device.performance_counters()

        if len(per_counters) != 0:
            fetcher = device.get_hwmon_fetcher()
            power_info = str(fetcher.read_powers_average()[0])
            p = round(float(power_info.split(" ")[-1]) / 1000000.0, 2)
            power = f"{p:.2f}"
            power_str = f"POWER({warboy_name}): {power}W"
            powers_str += "| {0:<21}".format(power_str)

        t_utils = 0.0
        for pc in per_counters:
            pe_name = str(pc[0])
            cur_pc = pc[1]

            if pe_name in last_pc:
                result = cur_pc.calculate_utilization(last_pc[pe_name])
                util = result.npu_utilization()
                if not ("0-1" in pe_name):
                    util /= 2.0
                t_utils += util

            last_pc[pe_name] = cur_pc
        if len(per_counters) != 0:
            t_utils = round(t_utils * 100.0, 2)
            util_str = f"Utilize({warboy_name}): {t_utils:.2f}%"
            utils_str += "| {0:<23}".format(util_str)

    return powers_str, utils_str


def make_full_grid(
    grid_imgs: List[np.ndarray],
    num_grid: int,
    grid_shape: Tuple[int, int],
    full_grid_shape: Tuple[int, int],
) -> np.ndarray:
    pad = int(full_grid_shape[0] * 0.1)
    full_grid = np.zeros(
        (full_grid_shape[0] + pad, full_grid_shape[1] + pad, 3), np.uint8
    )

    for i, grid_img in enumerate(grid_imgs):
        if grid_img is None:
            continue

        r = i // num_grid
        c = i % num_grid

        x0 = r * grid_shape[0] + (r + 1) * 5 + pad
        x1 = (r + 1) * grid_shape[0] + (r + 1) * 5 + pad
        y0 = c * grid_shape[1] + (c + 1) * 5
        y1 = (c + 1) * grid_shape[1] + (c + 1) * 5
        full_grid[x0:x1, y0:y1] = grid_img

    return full_grid


def put_warboy_info(
    img: np.ndarray, power_info: str, util_info: str, img_shape: Tuple[int, int]
):
    scale = max(int(0.0015 * img_shape[0]), 1)
    img = cv2.putText(
        img,
        power_info,
        (int(0.02 * img_shape[0]), int(0.01 * img_shape[1]) + 20),
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 255, 255),
        scale,
        cv2.LINE_AA,
    )
    img = cv2.putText(
        img,
        util_info,
        (int(0.02 * img_shape[0]), int(0.01 * img_shape[1]) + 45),
        cv2.FONT_HERSHEY_PLAIN,
        scale,
        (255, 255, 255),
        scale,
        cv2.LINE_AA,
    )
    return img
