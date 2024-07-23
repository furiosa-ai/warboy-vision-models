import cv2
import numpy as np
import pytest
import os, sys

import asyncio
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from pycocotools.cocoeval import COCOeval
from tests.test_config import TEST_MODEL_LIST, QUANTIZED_ONNX_DIR
from tests.utils import (
    MSCOCODataLoader,
    CONF_THRES,
    IOU_THRES,
    ANCHORS,
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    xyxy2xywh,
)
from warboy import WARBOY_YOLO
from warboy.utils.preprocess import YOLOPreProcessor
from warboy.utils.postprocess import ObjDetPostprocess

CONFIG_PATH = "./tests/test_config/object_detection"
YAML_PATH = [
    os.path.join(CONFIG_PATH, model_name + ".yaml")
    for model_name in TEST_MODEL_LIST["object_detection"]
]
WARBOY_MODELS = [WARBOY_YOLO(yaml) for yaml in YAML_PATH]

PARAMETERS = [
    (
        warboy.model_name,
        os.path.join(QUANTIZED_ONNX_DIR, "object_detection", warboy.onnx_i8_path),
        warboy.input_shape,
        warboy.anchors,
    )
    for warboy in WARBOY_MODELS
]

TARGET_ACCURACY = {
    "yolov5nu": 0.343,
    "yolov5su": 0.430,
    "yolov5mu": 0.490,
    "yolov5lu": 0.522,
    "yolov5xu": 0.532,
    "yolov5n": 0.280,
    "yolov5s": 0.374,
    "yolov5m": 0.454,
    "yolov5l": 0.490,
    "yolov5x": 0.507,
    "yolov7": 0.514,
    "yolov7x": 0.531,
    "yolov8n": 0.373,
    "yolov8s": 0.449,
    "yolov8m": 0.502,
    "yolov8l": 0.529,
    "yolov8x": 0.539,
    "yolov9t": 0.383,
    "yolov9s": 0.468,
    "yolov9m": 0.514,
    "yolov9c": 0.530,
    "yolov9e": 0.556,
}


async def warboy_inference(model, data_loader, preprocessor, postprocessor):
    async def task(
        pbar, runner, data_loader, preprocessor, postprocessor, worker_id, worker_num
    ):
        results = []
        for idx, (img_path, annotation) in enumerate(data_loader):
            if idx % worker_num != worker_id:
                continue

            img = cv2.imread(str(img_path))
            img0shape = img.shape[:2]
            input_, contexts = preprocessor(img, new_shape=data_loader.input_shape)
            preds = await runner.run([input_])

            outputs = postprocessor(preds, contexts, img0shape)[0]

            bboxes = xyxy2xywh(outputs[:, :4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2

            for output, bbox in zip(outputs, bboxes):
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                        "bbox": [round(x, 3) for x in bbox],
                        "score": round(output[4], 5),
                    }
                )
            pbar.update(1)
        return results

    from furiosa.runtime import create_runner

    worker_num = 16
    with tqdm(total=5000) as pbar:
        async with create_runner(model, worker_num=32) as runner:
            results = await asyncio.gather(
                *(
                    task(
                        pbar,
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
def test_warboy_yolo_accuracy_det(
    model_name: str, model: str, input_shape: List[int], anchors
):
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}

    preprocessor = YOLOPreProcessor()
    postprocessor = ObjDetPostprocess(
        model_name, model_cfg, None, False
    ).postprocess_func

    data_loader = MSCOCODataLoader(
        Path("/home/furiosa/work_space/val2017"),
        Path("/home/furiosa/work_space/annotations/instances_val2017.json"),
        preprocessor,
        input_shape,
    )

    results = asyncio.run(
        warboy_inference(model, data_loader, preprocessor, postprocessor)
    )

    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(coco_eval.stats[:3])
    assert coco_eval.stats[0] >= (
        TARGET_ACCURACY[model_name] * 0.95
    ), f"{model_name} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[model_name] * 0.95}]"
