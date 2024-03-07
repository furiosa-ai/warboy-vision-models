import glob
import os
import random

import cv2
from furiosa.optimizer import optimize_model
from furiosa.quantizer import (
    CalibrationMethod,
    Calibrator,
    ModelEditor,
    TensorType,
    get_pure_input_names,
    quantize,
)
import onnx
from tqdm import tqdm
import typer
import yaml

from utils.preprocess import preproc

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(cfg):
    onnx_path, input_shape, output_path, calib_data_path, num_data, method = get_params_from_cfg(
        cfg
    )

    model = onnx.load(onnx_path)
    model = optimize_model(model=model, opset_version=13, input_shapes={"images": input_shape})

    calib_data = glob.glob(calib_data_path + "/**", recursive=True)
    calib_data = random.choices(calib_data, k=num_data)

    calibrator = Calibrator(model, CalibrationMethod._member_map_[method])

    for data in tqdm(calib_data, desc="calibration"):
        if not (data.endswith(".png") or data.endswith(".jpg")):
            continue
        input_ = preproc(data, new_shape=(int(input_shape[2:][0]),int(input_shape[2:][1])))
        calibrator.collect_data([[input_]])

    ranges = calibrator.compute_range()

    ## Optimize Input Type
    editor = ModelEditor(model)
    input_names = get_pure_input_names(model)

    for input_name in input_names:
        editor.convert_input_type(input_name, TensorType.UINT8)

    quantized_model = quantize(model, ranges)

    with open(output_path, "wb") as f:
        f.write(bytes(quantized_model))

    print(f"Quantization completed >> {output_path}")
    return


def get_params_from_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg_info = yaml.full_load(f)

    model_info = cfg_info["model_info"]
    calib_info = cfg_info["quantization_info"]

    onnx_path = model_info["onnx_path"]
    input_shape = model_info["input_shape"]
    output_path = model_info["i8_onnx_path"]

    calib_data_path = calib_info["calib_data"]
    num_data = calib_info["num_data"]
    method = calib_info["method"]

    return onnx_path, input_shape, output_path, calib_data_path, num_data, method


if __name__ == "__main__":
    app()
