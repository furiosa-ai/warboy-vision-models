import os
import subprocess
from typing import Any, List
import yaml


def get_demo_params_from_cfg(cfg: str):
    """
    function for parsing demo parameters from config file (.yaml)
    """
    with open(cfg) as f:
        demo_params = list(yaml.load_all(f, Loader=yaml.FullLoader))[0]
        app_configs = demo_params["app_config"]    
        port = demo_params["port"]
        app_params = []
        for app_config in app_configs:
            runtime_params = []
            model_names = []
            input_shapes = []
            class_names = []
            
            for model_config in app_config["model_config"]:
                runtime_param, model_name, input_shape, class_name = (
                        get_model_params_from_cfg(model_config)
                    )
                runtime_params.append(runtime_param)
                model_names.append(model_name)
                input_shapes.append(input_shape)
                class_names.append(class_name)
            app_params.append(
                {
                    "app": app_config["application"],
                    "runtime_params": runtime_params,
                    "input_shape": input_shapes,
                    "class_names": class_names,
                    "model_name": model_names,
                    "model_path": app_config["model_path"],
                    "worker_num": int(app_config["num_worker"]),
                    "warboy_device": app_config["device"],
                    "videos_info": app_config["videos_info"],
                }
            ) 
    return app_params, port,


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