import os
from pathlib import Path
from typing import List

from pycocotools.cocoeval import COCOeval

from ...warboy import get_model_params_from_cfg
from ...warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine
from ...warboy.yolo.preprocess import YoloPreProcessor
from ..utils import (
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    MSCOCODataLoader,
    set_test_engin_configs,
    xyxy2xywh,
)

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
    "yolov7-w6": 0.549,
    "yolov7-e6": 0.560,
    "yolov7-d6": 0.566,
    "yolov7-e6e": 0.568,
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
    "yolov5n6": 0.360,
    "yolov5n6u": 0.421,
    "yolov5s6": 0.448,
    "yolov5s6u": 0.486,
    "yolov5m6": 0.513,
    "yolov5m6u": 0.536,
    "yolov5l6": 0.537,
    "yolov5l6u": 0.557,
    "yolov5x6": 0.550,
    "yolov5x6u": 0.568,
}


def _process_output(outputs_dict, data_loader):
    results = []
    for img_path, annotation in data_loader:
        if not len(outputs_dict[str(img_path)]) == 1:
            print(len(outputs_dict[str(img_path)]))
        for outputs in outputs_dict[str(img_path)]:
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
    return results


def test_warboy_yolo_accuracy_det(cfg: str, image_dir: str, annotation_file: str):
    """
    cfg(str): a path to config file
    image_dir(str): a path to image directory
    annotation_file(str): a path to annotation file
    """
    image_names = os.listdir(image_dir)

    images = [
        Image(image_info=os.path.join(image_dir, image_name))
        for image_name in image_names
    ]

    param = get_model_params_from_cfg(cfg)

    engin_configs = set_test_engin_configs(param, 2)

    preprocessor = YoloPreProcessor(new_shape=param["input_shape"], tensor_type="uint8")

    data_loader = MSCOCODataLoader(
        Path(image_dir),
        Path(annotation_file),
        preprocessor,
        param["input_shape"],
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
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(coco_eval.stats[:3])

    assert coco_eval.stats[0] >= (
        TARGET_ACCURACY[param["model_name"]] * 0.9
    ), f"{param['model_name']} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"

    print(
        f"{param['model_name']} Accuracy check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"
    )
