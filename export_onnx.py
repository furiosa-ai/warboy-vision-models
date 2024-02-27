import os
import sys

import torch
import typer
import yaml

from model_utils import load_onnx_extractor, load_torch_model
from model_utils.extractor import *

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(cfg):

    model_type, model_name, weight, onnx_path, _, input_shape, nc, num_anchors = (
        get_params_from_cfg(cfg)
    )

    model = load_torch_model(model_type, weight, model_name)
    assert model is not None, "Fail to load model!!!"

    model.eval()
    dummy_input = torch.zeros(*input_shape).to(torch.device("cpu"))

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
        model_type, model_name, nc, "images", input_shape, num_anchors
    )
    model = onnx.load(onnx_path)
    model = onnx_extractor(model)

    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_path)
    print(f"Fisnish creating onnx file!!! -> {onnx_path}")
    return


def get_params_from_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg_info = yaml.full_load(f)

    model_info = cfg_info["model_info"]
    return model_info.values()


if __name__ == "__main__":
    app()
