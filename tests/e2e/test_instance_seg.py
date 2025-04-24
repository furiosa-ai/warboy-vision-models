import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import pytest
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.postprocess import InsSegPostProcess
from src.warboy.yolo.preprocess import YoloPreProcessor
from tests.test_config import QUANTIZED_ONNX_DIR, TEST_MODEL_LIST
from tests.utils import (
    CONF_THRES,
    IOU_THRES,
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    MSCOCODataLoader,
    xyxy2xywh,
)

CONFIG_PATH = "./tests/test_config/instance_segmentation"
YAML_PATH = [
    os.path.join(CONFIG_PATH, model_name + ".yaml")
    for model_name in TEST_MODEL_LIST["instance_segmentation"]
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


TARGET_MASK_ACCURACY = {
    "yolov8n-seg": 0.305,
    "yolov8s-seg": 0.368,
    "yolov8m-seg": 0.408,
    "yolov8l-seg": 0.426,
    "yolov8x-seg": 0.434,
    "yolov9c-seg": 0.422,
    "yolov9e-seg": 0.443,
}

TARGET_BBOX_ACCURACY = {
    "yolov8n-seg": 0.367,
    "yolov8s-seg": 0.446,
    "yolov8m-seg": 0.499,
    "yolov8l-seg": 0.523,
    "yolov8x-seg": 0.534,
    "yolov9c-seg": 0.524,
    "yolov9e-seg": 0.551,
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

            outputs, pred_masks = postprocessor(preds, contexts, img0shape)[0]
            bboxes = xyxy2xywh(outputs[:, :4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2

            rles = [
                mask_util.encode(
                    np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in pred_masks
            ]
            for output, bbox, rle in zip(outputs, bboxes, rles):
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                        "bbox": [round(x, 3) for x in bbox],
                        "segmentation": rle,
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
def test_warboy_yolo_accuracy_seg(
    model_name: str, model: str, input_shape: List[int], anchors
):
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}
    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    postprocessor = InsSegPostProcess(
        model_name, model_cfg, None, False
    ).postprocess_func

    data_loader = MSCOCODataLoader(
        Path("datasets/coco/val2017"),
        Path("datasets/coco/annotations/instances_val2017.json"),
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
        coco_eval.stats[0] >= TARGET_MASK_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy (Mask) check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_MASK_ACCURACY[model_name] * 0.9}]"
    assert (
        coco_eval_box.stats[0] >= TARGET_BBOX_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy (Bbox) check failed! -> mAP: {coco_eval_box.stats[0]} [Target: {TARGET_BBOX_ACCURACY[model_name] * 0.9}]"
