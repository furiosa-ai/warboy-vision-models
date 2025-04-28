import os

import numpy as np
from furiosa.runtime.sync import create_runner

from warboy import get_model_params_from_cfg

from ..utils import TRACE_FILE_DIR


def test_warboy_performance(cfg, num_device):
    """
    model(str): a path to quantized onnx file
    num_device(int): a number of pe to use (1~2)    # CHECK
    """

    param = get_model_params_from_cfg(cfg)
    device = f"warboy({num_device})*1"
    from furiosa.runtime.profiler import profile

    input_shape = param["input_shape"]
    trace_dir = os.path.join(TRACE_FILE_DIR, param["task"])

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    trace_file = os.path.join(trace_dir, param["model_name"] + "_" + device + ".log")

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
