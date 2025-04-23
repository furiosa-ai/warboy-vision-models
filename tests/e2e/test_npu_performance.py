import os

import numpy as np
from furiosa.runtime.sync import create_runner

from src.warboy import get_model_params_from_cfg
from tests.test_config import TRACE_FILE_DIR


def set_engin_config(param, device):
    """
    FIXME
    get configs from config file
    currently, for yolov8n object detection model
    """
    engin_configs = []
    # for idx in 1:
    engin_config = {
        "name": f"test_{param['task']}_{0}",
        "task": param["task"],
        "model": param["onnx_i8_path"],
        "worker_num": 16,
        "device": device,
        "model_type": param["model_name"],
        "input_shape": param["input_shape"],
        "class_names": param["class_names"],
    }
    engin_configs.append(engin_config)
    return engin_configs


def test_warboy_performance(cfg, num_device):
    """
    model(str): a path to quantized onnx file
    num_device(int): a number of pe to use (1~2)    # CHECK
    """

    param = get_model_params_from_cfg(cfg)
    print(param)
    device = f"warboy({num_device})*1"
    engin_configs = set_engin_config(param, device)
    from furiosa.runtime.profiler import profile

    input_shape = engin_configs[0]["input_shape"]
    trace_dir = os.path.join(TRACE_FILE_DIR, engin_configs[0]["task"])

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    trace_file = os.path.join(
        trace_dir, engin_configs[0]["model_type"] + "_" + device + ".log"
    )

    onnx_i8_path = param["onnx_i8_path"]

    dummy_input = np.uint8(np.random.randn(*input_shape))

    with open(trace_file, mode="w") as tracing_file:
        with profile(file=tracing_file) as profiler:
            with create_runner(
                onnx_i8_path,
                device=device,
                compiler_config={"use_program_loading": True},
            ) as runner:
                for _ in range(30):
                    runner.run([dummy_input])

    return True
