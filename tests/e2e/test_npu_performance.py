import os

import numpy as np
import pytest
from furiosa.runtime.sync import create_runner

from src.warboy.cfg import get_model_params_from_cfg
from tests.test_config import (
    QUANTIZED_ONNX_DIR,
    TEST_MODEL_LIST,
    TEST_TASK,
    TRACE_FILE_DIR,
)

CONFIG_PATH = "./tests/test_config"
YAML_PATH = [
    os.path.join(CONFIG_PATH, task, model_name + ".yaml")
    for task in TEST_TASK
    for model_name in TEST_MODEL_LIST[task]
]
DEVICE = ["warboy(1)*1", "warboy(2)*1"]

PARAMS = [get_model_params_from_cfg(yaml) for yaml in YAML_PATH]

PARAMETERS = [
    (
        param["task"],
        param["model_name"],
        os.path.join(QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"]),
        param["input_shape"],
        device,
    )
    for param in PARAMS
    for device in DEVICE
]


@pytest.mark.parametrize("task, model_name, model, input_shape, device", PARAMETERS)
def test_warboy_performance(task, model_name, model, input_shape, device):
    from furiosa.runtime.profiler import profile

    trace_dir = os.path.join(TRACE_FILE_DIR, task)

    if "6" in model_name and device == "warboy(1)*1":
        return True

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    trace_file = os.path.join(trace_dir, model_name + "_" + device + ".log")
    dummy_input = np.uint8(np.random.randn(*input_shape))

    with open(trace_file, mode="w") as tracing_file:
        with profile(file=tracing_file) as profiler:
            with create_runner(
                model,
                device=device,
                compiler_config={"use_program_loading": True},
            ) as runner:
                for _ in range(30):
                    runner.run([dummy_input])

    assert True
