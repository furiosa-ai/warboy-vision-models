import glob
import os
import random
import sys

import cv2
import onnx
import typer
from tqdm import tqdm

HOME_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(HOME_DIR)
from furiosa.optimizer import optimize_model
from furiosa.quantizer import (
    CalibrationMethod,
    Calibrator,
    ModelEditor,
    TensorType,
    get_pure_input_names,
    quantize,
)

from utils.parse_params import get_model_params_from_cfg
from utils.preprocess import YOLOPreProcessor

app = typer.Typer(pretty_exceptions_show_locals=False)


def quantize_model(
    onnx_path: str, input_shape, output_path, calib_data_path, num_data, method
):
    model = onnx.load(onnx_path)
    model = optimize_model(
        model=model, opset_version=13, input_shapes={"images": [1, 3, *input_shape]}
    )

    calib_data = glob.glob(calib_data_path + "/**", recursive=True)
    calib_data = random.choices(calib_data, k=min(num_data, len(calib_data)))
    calibrator = Calibrator(model, CalibrationMethod._member_map_[method])

    preprocess = YOLOPreProcessor()
    for data in tqdm(calib_data, desc="calibration"):
        if not (data.endswith(".png") or data.endswith(".jpg")):
            continue
        img = cv2.imread(data)
        input_, _ = preprocess(
            img,
            new_shape=(int(input_shape[0]), int(input_shape[1])),
            tensor_type="float32",
        )
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


@app.command()
def main(cfg):
    params = get_model_params_from_cfg(cfg, mode="quantization")
    quantize_model(**params)
    return


if __name__ == "__main__":
    app()
