import os
import pytest
import numpy as np
from furiosa.runtime.sync import create_runner
from warboy.models import WARBOY_YOLO
from tests.test_config import (
    WEIGHT_DIR,
    ONNX_DIR,
    QUANTIZED_ONNX_DIR,
    TRACE_FILE_DIR,
    TEST_MODEL_LIST,
    TEST_TASK,
)


CONFIG_PATH = "./tests/test_config"
YAML_PATH = [
    os.path.join(CONFIG_PATH, task, model_name + ".yaml")
    for task in TEST_TASK
    for model_name in TEST_MODEL_LIST[task]
]
DEVICE = ["npu0pe0", "npu0pe0-1"]

PARAMETERS = [(WARBOY_YOLO(yaml), device) for yaml in YAML_PATH for device in DEVICE]


def test_warboy_performance(warboy_yolo, device):
    from furiosa.runtime.profiler import profile
    input_shape = warboy_yolo.input_shape
    trace_dir = os.path.join(TRACE_FILE_DIR, warboy_yolo.task)

    if "6" in warboy_yolo.model_name and device == "npu0pe0":
        return True 

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    trace_file = os.path.join(trace_dir, warboy_yolo.model_name + "_" + device + ".log")
    onnx_i8_path = os.path.join(
        QUANTIZED_ONNX_DIR, warboy_yolo.task, warboy_yolo.onnx_i8_path
    )
    dummy_input = np.uint8(np.random.randn(*input_shape))
    
    with open(trace_file, mode="w") as tracing_file:
        with profile(file=tracing_file) as profiler:
            with create_runner(
                onnx_i8_path, device=device, compiler_config={"use_program_loading": True}
            ) as runner:
                for _ in range(30):
                    runner.run([dummy_input])

    return True