from pathlib import Path
from typing import Callable, Dict, Iterator, Tuple

import numpy as np
from pycocotools.coco import COCO

TRACE_FILE_DIR = "../models/trace"

CONF_THRES = 0.001
IOU_THRES = 0.7

# CONF_THRES = 0.05
# IOU_THRES = 0.5

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
        self.input_shape = input_shape[2:]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict]]:
        for path in self.image_paths:
            yield path, self.image_filename_to_annotation[path.name]

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
