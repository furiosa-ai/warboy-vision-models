TASKS = {"object_detection", "pose_estimation", "instance_segmentation"}

MODEL_LIST = {
    "object_detection": [
        "yolov9t",
        "yolov9s",
        "yolov9m",
        "yolov9c",
        "yolov9e",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolov7",
        "yolov7x",
        "yolov7-w6",
        "yolov7-e6",
        "yolov7-d6",
        "yolov7-e6e",
        "yolov5n",
        "yolov5s",
        "yolov5m",
        "yolov5l",
        "yolov5x",
        "yolov5nu",
        "yolov5su",
        "yolov5mu",
        "yolov5lu",
        "yolov5xu",
        "yolov5n6u",
        "yolov5n6",
        "yolov5s6u",
        "yolov5s6",
        "yolov5m6u",
        "yolov5m6",
        "yolov5l6u",
        "yolov5l6",
        "yolov5x6u",
        "yolov5x6",
    ],
    "pose_estimation": [
        "yolov8n-pose",
        "yolov8s-pose",
        "yolov8m-pose",
        "yolov8l-pose",
        "yolov8x-pose",
    ],
    "instance_segmentation": [
        "yolov8n-seg",
        "yolov8s-seg",
        "yolov8m-seg",
        "yolov8l-seg",
        "yolov8x-seg",
        "yolov9c-seg",
        "yolov9e-seg",
    ],
}
import yaml
from typing import Dict, Any


def get_demo_params_from_cfg(cfg: str) -> Dict[str, Any]:
    """
    function for parsing demo parameters from config file (.yaml)

    args:
        cfg(str) : path of configuaration file (.yaml)
    """
    cfg_file = open(cfg)
    demo_config = list(yaml.load_all(cfg_file, Loader=yaml.FullLoader))[0]
    cfg_file.close()
    params = []
    for app_config in demo_config["app_config"]:
        param = {
            "task": app_config["task"][0],
            "model_path": app_config["model_path"][0],
            "worker_num": int(app_config["num_worker"]),
            "warboy_device": app_config["npu_device"][0],
            "videos_info": app_config["video_params"],
            "model_param": [],
            "class_name": [],
            "input_shape": [],
            "model_name": [],
        }
        param["model_param"], param["model_name"], param["input_shape"], param[
            "class_name"
        ] = _get_model_params_from_cfg(app_config["model_cfg"][0])

        params.append(param)

    return params


def _get_model_params_from_cfg(cfg: str) -> Dict[str, Any]:
    """
    function for parsing model configuration parameters from config file (.yaml)

    args:
        cfg(str) : path of configuaration file (.yaml)
    """
    cfg_file = open(cfg)
    model_cfg = yaml.full_load(cfg_file)
    cfg_file.close()

    param = [
        {
            "conf_thres": model_cfg["conf_thres"],
            "iou_thres": model_cfg["iou_thres"],
            "anchors": model_cfg["anchors"],
        },
        model_cfg["model_name"],
        model_cfg["input_shape"][2:],
        model_cfg["class_names"],
    ]
    return param


def get_model_params_from_cfg(cfg: str) -> Dict[str, Any]:
    """
    function for parsing model configuration parameters from config file (.yaml)

    args:
        cfg(str) : path of configuaration file (.yaml)
    """
    cfg_file = open(cfg)
    model_cfg = yaml.full_load(cfg_file)
    cfg_file.close()
    """
    add
    """
    return model_cfg


import numpy as np

COLORS = [
    (144, 238, 144),
    (255, 0, 0),
    (178, 34, 34),
    (221, 160, 221),
    (0, 255, 0),
    (0, 128, 0),
    (210, 105, 30),
    (220, 20, 60),
    (192, 192, 192),
    (255, 228, 196),
    (50, 205, 50),
    (139, 0, 139),
    (100, 149, 237),
    (138, 43, 226),
    (238, 130, 238),
    (255, 0, 255),
    (0, 100, 0),
    (127, 255, 0),
    (255, 0, 255),
    (0, 0, 205),
    (255, 140, 0),
    (255, 239, 213),
    (199, 21, 133),
    (124, 252, 0),
    (147, 112, 219),
    (106, 90, 205),
    (176, 196, 222),
    (65, 105, 225),
    (173, 255, 47),
    (255, 20, 147),
    (219, 112, 147),
    (186, 85, 211),
    (199, 21, 133),
    (148, 0, 211),
    (255, 99, 71),
    (144, 238, 144),
    (255, 255, 0),
    (230, 230, 250),
    (0, 0, 255),
    (128, 128, 0),
    (189, 183, 107),
    (255, 255, 224),
    (128, 128, 128),
    (105, 105, 105),
    (64, 224, 208),
    (205, 133, 63),
    (0, 128, 128),
    (72, 209, 204),
    (139, 69, 19),
    (255, 245, 238),
    (250, 240, 230),
    (152, 251, 152),
    (0, 255, 255),
    (135, 206, 235),
    (0, 191, 255),
    (176, 224, 230),
    (0, 250, 154),
    (245, 255, 250),
    (240, 230, 140),
    (245, 222, 179),
    (0, 139, 139),
    (143, 188, 143),
    (255, 0, 0),
    (240, 128, 128),
    (102, 205, 170),
    (60, 179, 113),
    (46, 139, 87),
    (165, 42, 42),
    (178, 34, 34),
    (175, 238, 238),
    (255, 248, 220),
    (218, 165, 32),
    (255, 250, 240),
    (253, 245, 230),
    (244, 164, 96),
    (210, 105, 30),
]
# PALETTE = np.array(
#     [
#         [255, 128, 0],
#         [255, 153, 51],
#         [255, 178, 102],
#         [230, 230, 0],
#         [255, 153, 255],
#         [153, 204, 255],
#         [255, 102, 255],
#         [255, 51, 255],
#         [102, 178, 255],
#         [51, 153, 255],
#         [255, 153, 153],
#         [255, 102, 102],
#         [255, 51, 51],
#         [153, 255, 153],
#         [102, 255, 102],
#         [51, 255, 51],
#         [0, 255, 0],
#         [0, 0, 255],
#         [255, 0, 0],
#         [255, 255, 255],
#     ],
#     np.int32,
# )
PALETTE = np.array(
    [
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
    ],
    np.int32,
)

SKELETONS = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

POSE_LIMB_COLOR = PALETTE[
    [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
].tolist()
