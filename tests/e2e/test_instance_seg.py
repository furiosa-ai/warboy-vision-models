import cv2
import numpy as np
import pytest
import os, sys

import asyncio
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

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
from warboy.utils.postprocess import InsSegPostProcess


CONFIG_PATH = "./tests/test_config/instance_segmentation"
YAML_PATH = [
    os.path.join(CONFIG_PATH, model_name + ".yaml")
    for model_name in TEST_MODEL_LIST["instance_segmentation"]
]
WARBOY_MODELS = [WARBOY_YOLO(yaml) for yaml in YAML_PATH]

PARAMETERS = [
    (
        warboy.model_name,
        os.path.join(QUANTIZED_ONNX_DIR, "instance_segmentation", warboy.onnx_i8_path),
        warboy.input_shape,
        warboy.anchors,
    )
    for warboy in WARBOY_MODELS
]

TARGET_MASK_ACCURACY = {
    "yolov8n-pose": 0.305,
    "yolov8s-pose": 0.368,
    "yolov8m-pose": 0.408,
    "yolov8l-pose": 0.426,
    "yolov8x-pose": 0.434,
}

TARGET_BBOX_ACCURACY = {
    "yolov8n-pose": 0.367,
    "yolov8s-pose": 0.446,
    "yolov8m-pose": 0.499,
    "yolov8l-pose": 0.523,
    "yolov8x-pose": 0.534,
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

            outputs, pred_masks = postprocessor(preds, contexts, img0shape)[0]
            bboxes = xyxy2xywh(outputs[:, :4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2

            rles = [
                mask_util.encode(
                    np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in pred_masks
            ]
            for (output, bbox, rle) in zip(outputs, bboxes, rles):
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                        "bbox": [round(x, 3) for x in bbox],
                        "segmentation": rle,
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
def test_warboy_yolo_accuracy_pose(
    model_name: str, model: str, input_shape: List[int], anchors
):
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}
    preprocessor = YOLOPreProcessor()
    postprocessor = InsSegPostProcess(
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
    coco_eval = COCOeval(data_loader.coco, coco_result, "segm")
    coco_eval_box = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    coco_eval_box.summarize()

    assert (
        coco_eval.stats[0] >= TARGET_MASK_ACCURACY[warboy.model_name]
    ), f"{model_name} Accuracy (Mask) check failed! -> mAP: {coco_eval.stats[0]}"
    assert (
        coco_eval_box.stats[0] >= TARGET_BBOX_ACCURACY[warboy.model_name]
    ), f"{model_name} Accuracy (Bbox) check failed! -> mAP: {coco_eval_box.stats[0]}"
