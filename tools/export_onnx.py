import os
import sys
from typing import Tuple

import torch
import typer
import yaml

HOME_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(HOME_DIR)
from tools.model_utils import load_onnx_extractor, load_torch_model
from tools.model_utils.extractor import *
from utils.parse_params import get_model_params_from_cfg

app = typer.Typer(pretty_exceptions_show_locals=False)


def export_onnx_file(
    application: str,
    model_name: str,
    weight: str,
    onnx_path: str,
    input_shape: Tuple[int, int],
    num_classes: int,
    num_anchors: int,
):
    model = load_torch_model(application, weight, model_name)
    assert model is not None, "Fail to load model!!!"

    model.eval()
    dummy_input = torch.zeros(1, 3, *input_shape).to(torch.device("cpu"))

    print("Start creating onnx file....")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["images"],
        output_names=["outputs"],
    )
    onnx_extractor = load_onnx_extractor(
        application,
        model_name,
        num_classes,
        "images",
        [1, 3, *input_shape],
        num_anchors,
    )
    model = onnx.load(onnx_path)
    model = onnx_extractor(model)

    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_path)
    print(f"Creating onnx file done! -> {onnx_path}")
    return


@app.command()
def main(cfg):
    params = get_model_params_from_cfg(cfg, mode="export_onnx")
    del sys.modules["utils"]
    export_onnx_file(**params)
    return


def get_params_from_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg_info = yaml.full_load(f)

    model_info = cfg_info["model_info"]
    return model_info.values()


if __name__ == "__main__":
    app()
