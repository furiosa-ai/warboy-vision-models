import os
from pathlib import Path
from typing import List

import numpy as np
import pycocotools.mask as mask_util
import pytest
from pycocotools.cocoeval import COCOeval

from tests.utils import (
    CONF_THRES,
    IOU_THRES,
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    MSCOCODataLoader,
    xyxy2xywh,
)
from warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine
from warboy.yolo.preprocess import YoloPreProcessor

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


def set_engin_config(num_device, model, model_name, input_shape):
    """
    FIXME
    get configs from config file
    currently, for yolov8n instance segmentation model
    """
    engin_configs = []
    for idx in range(num_device):
        engin_config = {
            "name": f"test{idx}",
            "task": "instance_segmentation",
            "model": model,
            "worker_num": 16,
            "device": "warboy(1)*1",
            "model_type": model_name,
            "input_shape": input_shape,
            "class_names": [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ],
            "conf_thres": CONF_THRES,
            "iou_thres": IOU_THRES,
            "use_tracking": False,
        }
        engin_configs.append(engin_config)
    return engin_configs


def _process_output(outputs_dict, data_loader):
    results = []
    for img_path, annotation in data_loader:
        if not len(outputs_dict[str(img_path)]) == 1:
            print(len(outputs_dict[str(img_path)]))
        for outputs, pred_masks in outputs_dict[str(img_path)]:
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


@pytest.mark.parametrize("model_name, model, input_shape, anchors")
def test_warboy_yolo_accuracy_seg(
    model_name: str, model: str, input_shape: List[int], anchors
):
    import time

    t1 = time.time()

    image_dir = "datasets/coco/val2017"
    image_names = os.listdir(image_dir)

    images = [
        Image(image_info=os.path.join(image_dir, image_name))
        for image_name in image_names
    ]

    engin_configs = set_engin_config(2, model, model_name, input_shape[2:])

    preprocessor = YoloPreProcessor(new_shape=input_shape, tensor_type="uint8")

    data_loader = MSCOCODataLoader(
        Path("datasets/coco/val2017"),
        Path("datasets/coco/annotations/instances_val2017.json"),
        preprocessor,
        input_shape,
    )

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
    results = _process_output(outputs, data_loader)

    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "segm")
    coco_eval_box = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    coco_eval_box.summarize()

    t2 = time.time()

    print(t2 - t1)

    print("MASK mAP: ", coco_eval.stats[0])
    print("BBOX mAP: ", coco_eval_box.stats[0])

    assert coco_eval.stats[0] >= (
        TARGET_MASK_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy (Mask) check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_MASK_ACCURACY[model_name] * 0.9}]"

    print(
        f"{model_name} Accuracy (Mask) check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_MASK_ACCURACY[model_name] * 0.9}]"
    )


    assert coco_eval.stats[0] >= (
        TARGET_MASK_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy (Bbox) check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_BBOX_ACCURACY[model_name] * 0.9}]"

    print(
        f"{model_name} Accuracy (Bbox) check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_BBOX_ACCURACY[model_name] * 0.9}]"
    )
