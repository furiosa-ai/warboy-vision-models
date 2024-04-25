import os
import subprocess
from typing import Any, List

import yaml


def get_demo_params_from_cfg(cfg: str):
    """
    function for parsing demo parameters from config file (.yaml)
    """
    num_channel = 0
    with open(cfg) as f:
        demo_params = yaml.load_all(f, Loader=yaml.FullLoader)
        params = []
        for demo_param in demo_params:
            runtime_params, model_name, input_shape, class_names = (
                get_model_params_from_cfg(demo_param["model_config"])
            )
            if os.path.exists(demo_param["output_path"]):
                subprocess.run(["rm", "-rf", demo_param["output_path"]])
            os.makedirs(demo_param["output_path"])
            params.append(
                {
                    "app": demo_param["application"],
                    "runtime_params": runtime_params,
                    "input_shape": input_shape,
                    "class_names": class_names,
                    "model_name": model_name,
                    "model_path": demo_param["model_path"],
                    "worker_num": int(demo_param["num_worker"]),
                    "warboy_device": demo_param["device"],
                    "video_paths": demo_param["video_path"],
                    "output_path": demo_param["output_path"],
                }
            )
    return params


def get_model_params_from_cfg(cfg: str, mode: str = "runtime") -> List[Any]:
    """
    function for parsing export_onnx.py, furiosa_quantizer.py and runtime parameters from config file (.yaml)
    """
    f = open(cfg)
    model_params = cfg_info = yaml.full_load(f)
    f.close()

    application = model_params["application"]
    model_name = model_params["model_name"]
    weight = model_params["weight"]
    onnx_path = model_params["onnx_path"]
    onnx_i8_path = model_params["onnx_i8_path"]

    calibration_method, calibration_data, num_calibration_data = model_params[
        "calibration_params"
    ].values()

    input_shape = model_params["input_shape"]
    class_names = model_params["class_names"]
    anchors = model_params["anchors"]
    num_classes = len(class_names)
    num_anchors = 3 if anchors[0] is None else len(anchors)
    conf_thres = model_params["conf_thres"]
    iou_thres = model_params["iou_thres"]
    params = []

    if mode == "export_onnx":
        params = {
            "application": application,
            "model_name": model_name,
            "weight": weight,
            "onnx_path": onnx_path,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "num_anchors": num_anchors,
        }
    elif mode == "quantization":
        params = {
            "onnx_path": onnx_path,
            "input_shape": input_shape,
            "output_path": onnx_i8_path,
            "calib_data_path": calibration_data,
            "num_data": num_calibration_data,
            "method": calibration_method,
        }
    elif mode == "inference":
        params = [
            {"conf_thres": conf_thres, "iou_thres": iou_thres, "anchors": anchors},
            application,
            model_name,
            onnx_i8_path,
            input_shape,
            class_names,
        ]
    else:
        params = [
            {"conf_thres": conf_thres, "iou_thres": iou_thres, "anchors": anchors},
            model_name,
            input_shape,
            class_names,
        ]

    return params


def get_output_paths(cfg):
    output_paths = []
    with open(cfg) as f:
        demo_params = yaml.load_all(f, Loader=yaml.FullLoader)
        for param in demo_params:
            for video_path in param["video_path"]:
                video_name = (video_path.split("/")[-1]).split(".")[0]
                output_paths.append(
                    os.path.join(param["output_path"], "_output", video_name)
                )
    return output_paths
