import os
from pathlib import Path
from typing import List

from pycocotools.cocoeval import COCOeval

from ...warboy import get_model_params_from_cfg
from ...warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine
from ...warboy.yolo.preprocess import YoloPreProcessor
from ..utils import MSCOCODataLoader, set_test_engin_configs

TARGET_ACCURACY = {
    "yolov8n-pose": 0.504,
    "yolov8s-pose": 0.600,
    "yolov8m-pose": 0.650,
    "yolov8l-pose": 0.676,
    "yolov8x-pose": 0.692,
}


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


def test_warboy_yolo_accuracy_pose(cfg: str, image_dir: str, annotation_file: str):
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
    coco_eval = COCOeval(data_loader.coco, coco_result, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(coco_eval.stats[:3])

    if coco_eval.stats[0] >= (TARGET_ACCURACY[param["model_name"]] * 0.9):
        print(
            f"{param['model_name']} Accuracy check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"
        )

    else:
        print(
            f"{param['model_name']} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"
        )
