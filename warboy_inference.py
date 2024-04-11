import os
import subprocess

import cv2
import numpy as np
import typer
import yaml
from furiosa.runtime.sync import create_runner

from utils.parse_params import get_model_params_from_cfg
from utils.postprocess import getPostProcesser
from utils.preprocess import YOLOPreProcessor, letterbox

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(cfg, input_path):
    runner_info, app_type, model_name, model_path, input_shape, class_names = (
        get_model_params_from_cfg(cfg, mode="inference")
    )

    result_path = "output"
    if os.path.exists(result_path):
        subprocess.run(["rm", "-rf", result_path])
    os.makedirs(result_path)

    preprocessor = YOLOPreProcessor()
    postprocessor = getPostProcesser(app_type, model_name, runner_info, class_names)

    input_images = os.listdir(str(input_path))

    with create_runner(model_path) as runner:
        for input_img in input_images:
            img = cv2.imread(os.path.join(input_path, input_img))
            input_, preproc_params = preprocessor(img, new_shape=input_shape)
            output = runner.run([input_])
            out = postprocessor(output, preproc_params, img)
            cv2.imwrite(os.path.join(result_path, input_img), out)


if __name__ == "__main__":
    app()
