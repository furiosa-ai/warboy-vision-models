import os
from pathlib import Path
from typing import List

import pytest
from pycocotools.cocoeval import COCOeval

from warboy_vision_models.tests.utils import CONF_THRES, IOU_THRES, MSCOCODataLoader
from warboy_vision_models.warboy.utils.process_pipeline import (
    Engine,
    Image,
    ImageList,
    PipeLine,
)
from warboy_vision_models.warboy.yolo.preprocess import YoloPreProcessor

TARGET_ACCURACY = {
    "yolov8n-pose": 0.504,
    "yolov8s-pose": 0.600,
    "yolov8m-pose": 0.650,
    "yolov8l-pose": 0.676,
    "yolov8x-pose": 0.692,
}


def set_engin_config(num_device, model, model_name, input_shape):
    """
    FIXME
    get configs from config file
    currently, for yolov8n pose estimation model
    """
    engin_configs = []
    for idx in range(num_device):
        engin_config = {
            "name": f"test{idx}",
            "task": "pose_estimation",
            "model": model,
            "worker_num": 16,
            "device": "warboy(1)*1",
            "model_type": model_name,
            "input_shape": input_shape,
            "class_names": ["person"],
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
        for outputs in outputs_dict[str(img_path)]:
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


@pytest.mark.parametrize("model_name, model, input_shape, anchors")
def test_warboy_yolo_accuracy_pose(
    model_name: str, model: str, input_shape: List[int], anchors
):
    """
    model_name(str):
    model(str): a path to quantized onnx file
    input_shape(List[int]): [N, C, H, W] => consider batch as 1
    anchors(List): [None] for yolov8
    """
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
        Path("datasets/coco/annotations/person_keypoints_val2017.json"),
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

    outputs = task.outputs
    results = _process_output(outputs, data_loader)

    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    t2 = time.time()

    print(t2 - t1)

    print(coco_eval.stats[:3])

    assert coco_eval.stats[0] >= (
        TARGET_ACCURACY[model_name] * 0.95
    ), f"{model_name} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[model_name] * 0.95}]"

    print(
        f"{model_name} Accuracy check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[model_name] * 0.95}]"
    )
