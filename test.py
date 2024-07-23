import os
import pytest

from warboy import WARBOY_YOLO
from tests.test_config import (
    WEIGHT_DIR,
    ONNX_DIR,
    QUANTIZED_ONNX_DIR,
    TEST_MODEL_LIST,
    TEST_TASK,
)


CONFIG_PATH = "./tests/test_config"
YAML_PATH = [
    os.path.join(CONFIG_PATH, task, model_name + ".yaml")
    for task in TEST_TASK
    for model_name in TEST_MODEL_LIST[task]
]
PARAMETERS = [WARBOY_YOLO(yaml) for yaml in YAML_PATH]


def test_export_onnx(warboy_yolo):
    task = warboy_yolo.task
    warboy_yolo.weight = os.path.join(WEIGHT_DIR, task, warboy_yolo.weight)
    warboy_yolo.onnx_path = os.path.join(ONNX_DIR, task, warboy_yolo.onnx_path)
    if os.path.exists(warboy_yolo.onnx_path):
        return
    assert (
        warboy_yolo.export_onnx()
    ), f"{warboy_yolo.model_name} -> ONNX Export Failed!!"


def test_furiosa_quantizer(warboy_yolo):
    task = warboy_yolo.task
    warboy_yolo.onnx_path = os.path.join(ONNX_DIR, task, warboy_yolo.onnx_path)
    if os.path.exists(warboy_yolo.onnx_i8_path):
        return
    warboy_yolo.onnx_i8_path = os.path.join(
        QUANTIZED_ONNX_DIR, task, warboy_yolo.onnx_i8_path
    )
    assert (
        warboy_yolo.quantize()
    ), f"{warboy_yolo.model_name} -> Furiosa Quantization Failed!!"


warboy_yolo = WARBOY_YOLO("./tests/test_config/object_detection/yolov5s6u.yaml")
test_export_onnx(warboy_yolo)
# test_furiosa_quantizer(warboy_yolo)
