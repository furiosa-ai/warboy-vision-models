import os, sys
import cv2
import subprocess
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Any

from furiosa.runtime.profiler import profile
from furiosa.runtime.sync import create_runner

HOME_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(HOME_DIR)
from tools.export_onnx import export_onnx_file
from tools.furiosa_quantizer import quantize_model
from utils_.preprocess import YOLOPreProcessor
from utils_.postprocess_func.output_decoder import (
    ObjDetDecoder,
    PoseEstDecoder,
    InsSegDecoder,
)

ANCHORS = {
    "yolov8": None,
    "yolov7": [
        [12, 16, 19, 36, 40, 28],
        [36, 75, 76, 55, 72, 146],
        [142, 110, 192, 243, 459, 401],
    ],
    "yolov7_6": [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ],
    "yolov5": [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ],
    "yolov5_6": [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ],
}

# Model list
'''
        "yolov8n": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov8"],
            "num_anchors": 3,
        },
        "yolov8s": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov8"],
            "num_anchors": 3,
        },
        "yolov8m": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov8"],
            "num_anchors": 3,
        },
        "yolov8l": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov8"],
            "num_anchors": 3,
        },
        
        "yolov8x": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov8"],
            "num_anchors": 3,
        },
        '''

MODEL_LIST = {
    "object_detection": {
        "yolov7": {
            "input_shape": [640, 640],
            "anchors": ANCHORS["yolov7"],
            "num_anchors": 3,
        },
        "yolov7x": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov7"],
            "num_anchors": 3,
        },
        "yolov7-w6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov7_6"],
            "num_anchors": 4,
        },
        "yolov7-e6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov7_6"],
            "num_anchors": 4,
        },
        "yolov7-d6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov7_6"],
            "num_anchors": 4,
        },
        "yolov7-e6e": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov7_6"],
            "num_anchors": 4,
        },
        "yolov5n": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov5"],
            "num_anchors": 3,
        },
        "yolov5s": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov5"],
            "num_anchors": 3,
        },
        "yolov5m": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov5"],
            "num_anchors": 3,
        },
        "yolov5l": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov5"],
            "num_anchors": 3,
        },
        "yolov5x": {
            "input_shape": [ 640, 640],
            "anchors": ANCHORS["yolov5"],
            "num_anchors": 3,
        },
        "yolov5n6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov5_6"],
            "num_anchors": 4,
        },
        "yolov5s6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov5_6"],
            "num_anchors": 4,
        },
        "yolov5m6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov5_6"],
            "num_anchors": 4,
        },
        "yolov5l6": {
            "input_shape": [ 1280, 1280],
            "anchors": ANCHORS["yolov5_6"],
            "num_anchors": 4,
        },
    },
    "pose_estimation": {},
    "instance_segmentation": {},
}

YOLO_CATEGORY_TO_COCO_CATEGORY = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


class MSCOCODataLoader:
    """Data loader for MSCOCO dataset"""

    def __init__(
        self, image_dir: Path, annotations_file: Path, preprocess: Callable, input_shape
    ) -> None:
        self.coco = COCO(annotations_file)
        coco_images = self.coco.dataset["images"]
        self.image_paths = list(image_dir / image["file_name"] for image in coco_images)
        self.image_filename_to_annotation = {
            image["file_name"]: image for image in coco_images
        }
        self.preprocess = preprocess
        self.input_shape = input_shape

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict]]:
        for path in self.image_paths:
            img = cv2.imread(str(path))
            img0shape = img.shape[:2]
            yield self.preprocess(
                img, new_shape=self.input_shape
            ), self.image_filename_to_annotation[path.name], img0shape

    def __len__(self) -> int:
        return len(self.image_paths)


# https://github.com/ultralytics/yolov5/blob/v6.2/utils/general.py#L703-L710
def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    # pylint: disable=invalid-name
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def warboy_test(
    log_file,
    trace_file,
    model_name,
    onnx_i8_path,
    input_shape,
    anchors,
    data_path,
    anno_path,
    device,
    mode="test"
):
    conf_thres = 0.001
    iou_thres = 0.7

    preprocessor = YOLOPreProcessor()
    postprocess_func = ObjDetDecoder(
        model_name, conf_thres, iou_thres, anchors, use_tracker=False
    )

    dummy_input = np.uint8(np.random.rand(1, 3, *input_shape))
    data_loader = MSCOCODataLoader(
        Path(data_path), Path(anno_path), preprocessor, input_shape
    )
    ## Performance Test
    with open(trace_file, "w") as output:
        with profile(file=output) as profiler:
            with create_runner(onnx_i8_path, device=device) as runner:
                with profiler.record("trace") as record:
                    for _ in range(0, 30):
                        runner.run([dummy_input])

    if mode != "test":
        return

    results = []
    # Accuracy Test
    with create_runner(onnx_i8_path, device=device) as runner:
        for (input_data, shapes), annotation, img0shape in tqdm(
            data_loader,
            desc="Evaluating",
            unit="image",
            mininterval=0.5,
            total=len(data_loader),
        ):
            outputs = runner.run([input_data])
            predictions = postprocess_func(outputs, shapes, img0shape)[0]

            boxes = xyxy2xywh(predictions[:, :4])
            boxes[:, :2] -= boxes[:, 2:] / 2
            for prediction, box in zip(predictions, boxes):
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[
                            int(prediction[5])
                        ],
                        "bbox": [round(x, 3) for x in box],
                        "score": round(prediction[4], 5),
                    }
                )

        coco_detections = data_loader.coco.loadRes(results)
        coco_eval = COCOeval(data_loader.coco, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP_50_95, mAP_50, mAP_75 = coco_eval.stats[:3]

        with open(log_file, mode="a") as log_file:
            log_file.write(
                f"{model_name}, {input_shape[2]}, {mAP_50_95}, {mAP_50}, {mAP_75}\n"
            )
    return


# Project Test for Object Detection Models
def object_detection_test(data_path, anno_path):
    model_list = MODEL_LIST["object_detection"]

    """
    Output Files 
    - warboy-vision-models/result/object_detection/onnx_files : ONNX Files 
    - warboy-vision-models/result/object_detection/traces : tracing (.json) files 
    - warboy-vision-models/result/object_detection/accuracy.log : log file for accuracy of models
    """

    result_path = os.path.join(HOME_DIR, "result")
    onnx_dir_path = os.path.join(result_path, "object_detection", "onnx_files")
    trace_dir_path = os.path.join(result_path, "object_detection", "traces")
    accuracy_log = os.path.join(result_path, "object_detection", "accuracy.log")
    '''
    if os.path.exists(onnx_dir_path):
        subprocess.run(["rm", "-rf", onnx_dir_path])

    if os.path.exists(trace_dir_path):
        subprocess.run(["rm", "-rf", trace_dir_path])

    os.makedirs(onnx_dir_path)
    os.makedirs(trace_dir_path)
    '''
    for model_name in model_list:
        model_info = model_list[model_name]
        input_shape = model_info["input_shape"]
        num_classes = 80  # COCO Category
        anchors = model_info["anchors"]
        num_anchors = model_info["num_anchors"]

        weight_file = os.path.join(HOME_DIR, "weights", model_name + ".pt")

        # 1. Export Onnx from torch model
        onnx_path = os.path.join(onnx_dir_path, model_name + ".onnx")
        export_onnx_file(
            "object_detection",
            model_name,
            weight_file,
            onnx_path,
            input_shape,
            num_classes,
            num_anchors,
        )

        # 2. Quantization using Furiosa SDK
        onnx_i8_path = os.path.join(onnx_dir_path, model_name + "_i8.onnx")
        num_calidataion_data = 200
        quantize_model(
            onnx_path,
            input_shape,
            onnx_i8_path,
            data_path,
            num_calidataion_data,
            "SQNR_ASYM",
        )

        # 3. Inference using Furiosa SDK Runtime (Accuracy Test)
        trace_path = os.path.join(trace_dir_path, "tracing_" + model_name + ".json")
        trace_single_path = os.path.join(
            trace_dir_path, "tracing_single_" + model_name + ".json"
        )
        try:
            warboy_test(
                accuracy_log,
                trace_path,
                model_name,
                onnx_i8_path,
                input_shape,
                anchors,
                data_path,
                anno_path,
                "npu0pe0-1",
            )
        except Exception as e:
            with open(accuracy_log, mode="a") as log_file:
                log_file.write(f"{model_name} -> Compile Fail! (Fusion)\n")
        try:
            warboy_test(
                accuracy_log,
                trace_single_path,
                model_name,
                onnx_i8_path,
                input_shape,
                anchors,
                data_path,
                anno_path,
                "npu0pe0",
                "not-test"
            )
        except Exception as e:
            with open(accuracy_log, mode="a") as log_file:
                log_file.write(f"{model_name} -> Compile Fail! (Single PE)\n")

def pose_estimation_test(data_path, anno_path):
    return


def instance_segmentation_test(data_path, anno_path):
    return


object_detection_test(
    "/home/ubuntu/.jongwook/datas/val2017",
    "/home/ubuntu/.jongwook/test/annotations/instances_val2017.json",
)
