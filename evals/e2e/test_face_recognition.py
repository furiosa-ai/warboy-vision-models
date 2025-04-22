import os
from pathlib import Path
from typing import List

import numpy as np
import pycocotools.mask as mask_util
import pytest
import typer
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity

from evals.utils import CONF_THRES, IOU_THRES
from warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine


def set_engin_config(num_device, model, model_name, input_shape):
    """
    FIXME
    get configs from config file
    currently, for facenet face recognition model
    """
    engin_configs = []
    for idx in range(num_device):
        engin_config = {
            "name": f"test{idx}",
            "task": "face_recognition",
            "model": model,
            "worker_num": 16,
            "device": "warboy(1)*1",
            "model_type": model_name,
            "input_shape": input_shape,
            "class_names": [],
            "conf_thres": CONF_THRES,
            "iou_thres": IOU_THRES,
            "use_tracking": False,
        }
        engin_configs.append(engin_config)
    return engin_configs


def _cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th).squeeze()
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def _test_performance(lfw_data_dir, pair_list, fe_dict):
    with open(pair_list, "r") as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        key_1, key_2 = os.path.join(lfw_data_dir, splits[0]), os.path.join(
            lfw_data_dir, splits[1]
        )
        if key_1 not in fe_dict.keys() or key_2 not in fe_dict.keys():
            continue
        fe_1 = fe_dict[key_1][0]
        fe_2 = fe_dict[key_2][0]
        label = int(splits[2])
        sim = cosine_similarity(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    print(len(fe_dict), len(labels))
    acc, th = _cal_accuracy(sims, labels)
    return acc, th


def _resolve_input_paths(input_path: Path) -> List[str]:
    """Create input file list"""
    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory may contain image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        return [
            str(p.resolve())
            for p in input_path.glob("**/*")
            if p.suffix.lower() in image_extensions
        ]
    else:
        typer.echo(f"Invalid input path '{str(input_path)}'")
        raise typer.Exit(1)


@pytest.mark.parametrize("model_name, model, input_shape, anchors")
def test_warboy_facenet_accuracy_recog(
    model_name: str, model: str, input_shape: List[int], anchors
):
    import time

    t1 = time.time()

    image_dir = "datasets/face_recognition/lfw-align-128"
    lfw_test_pair = "datasets/face_recognition/lfw_test_pair.txt"
    image_paths = _resolve_input_paths(Path(image_dir))

    images = [Image(image_info=image_path) for image_path in image_paths]

    engin_configs = set_engin_config(2, model, model_name, input_shape[2:])

    task = PipeLine(run_fast_api=False, run_e2e_test=True, num_channels=len(images))

    for idx, engin in enumerate(engin_configs):
        task.add(Engine(**engin), postprocess_as_img=False)
        task.add(
            ImageList(
                image_list=[image for image in images[idx :: len(engin_configs)]]
            ),
            name=engin["name"],
            postprocess_as_img=False,
        )

    # task.run(runtime_type="application")
    task.run()

    print("Inference done!")
    outputs = task.outputs

    acc, th = _test_performance(image_dir, lfw_test_pair, outputs)

    t2 = time.time()

    print(t2 - t1)

    print("Accuracy: ", acc)
    print("Threshold: ", th)
