import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.postprocess import PoseEstPostprocess
from src.warboy.yolo.preprocess import YoloPreProcessor
from tests.test_config import QUANTIZED_ONNX_DIR, TEST_MODEL_LIST
from tests.utils import CONF_THRES, IOU_THRES, MSCOCODataLoader

CONFIG_PATH = "./tests/test_config/pose_estimation"
YAML_PATH = [
    os.path.join(CONFIG_PATH, model_name + ".yaml")
    for model_name in TEST_MODEL_LIST["pose_estimation"]
]
PARAMS = [get_model_params_from_cfg(yaml) for yaml in YAML_PATH]

PARAMETERS = [
    (
        param["model_name"],
        os.path.join(QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"]),
        param["input_shape"],
        param["anchors"],
    )
    for param in PARAMS
]

TARGET_ACCURACY = {
    "yolov8n-pose": 0.504,
    "yolov8s-pose": 0.600,
    "yolov8m-pose": 0.650,
    "yolov8l-pose": 0.676,
    "yolov8x-pose": 0.692,
}


async def warboy_inference(model, data_loader, preprocessor, postprocessor):
    async def task(
        runner, data_loader, preprocessor, postprocessor, worker_id, worker_num
    ):
        results = []
        for idx, (img_path, annotation) in enumerate(data_loader):
            if idx % worker_num != worker_id:
                continue

            img = cv2.imread(str(img_path))
            img0shape = img.shape[:2]
            input_, contexts = preprocessor(img)
            preds = await runner.run([input_])

            outputs = postprocessor(preds, contexts, img0shape)[0]

            for output in outputs:
                keypoint = output[5:]
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": 1,
                        "keypoints": keypoint,
                        "score": round(output[4], 5),
                    }
                )
        return results

    from furiosa.runtime import create_runner

    worker_num = 16
    async with create_runner(
        model, worker_num=32, compiler_config={"use_program_loading": True}
    ) as runner:
        results = await asyncio.gather(
            *(
                task(
                    runner,
                    data_loader,
                    preprocessor,
                    postprocessor,
                    idx,
                    worker_num,
                )
                for idx in range(worker_num)
            )
        )
    return sum(results, [])


@pytest.mark.parametrize("model_name, model, input_shape, anchors", PARAMETERS)
def test_warboy_yolo_accuracy_pose(
    model_name: str, model: str, input_shape: List[int], anchors
):
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}

    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    postprocessor = PoseEstPostprocess(
        model_name, model_cfg, None, False
    ).postprocess_func

    data_loader = MSCOCODataLoader(
        Path("datasets/coco/val2017"),
        Path("datasets/coco/annotations/person_keypoints_val2017.json"),
        preprocessor,
        input_shape,
    )

    results = asyncio.run(
        warboy_inference(model, data_loader, preprocessor, postprocessor)
    )

    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    assert coco_eval.stats[0] >= (
        TARGET_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[model_name] * 0.9}]"
