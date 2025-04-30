import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
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


def test_warboy_yolo_accuracy_seg(cfg: str, image_dir: str, annotation_file: str):
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

    print("MASK mAP: ", coco_eval.stats[0])
    print("BBOX mAP: ", coco_eval_box.stats[0])

    if coco_eval.stats[0] >= (TARGET_MASK_ACCURACY[param["model_name"]] * 0.9):
        print(
            f"{param['model_name']} Accuracy (Mask) check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_MASK_ACCURACY[param['model_name']] * 0.9}]"
        )

    else:
        print(
            f"{param['model_name']} Accuracy (Mask) check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_MASK_ACCURACY[param['model_name']] * 0.9}]"
        )

    if coco_eval.stats[0] >= (TARGET_BBOX_ACCURACY[param["model_name"]] * 0.9):
        print(
            f"{param['model_name']} Accuracy (Bbox) check success! -> mAP: {coco_eval_box.stats[0]} [Target: {TARGET_BBOX_ACCURACY[param['model_name']] * 0.9}]"
        )

    else:
        print(
            f"{param['model_name']} Accuracy (Bbox) check failed! -> mAP: {coco_eval_box.stats[0]} [Target: {TARGET_BBOX_ACCURACY[param['model_name']] * 0.9}]"
        )
