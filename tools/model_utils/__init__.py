from typing import Any, List

from .extractor import YOLO_ONNX_Extractor as YOLO_ONNX_Extractor
from .object_detection_models import ObjDet_YOLO_Extractor, load_od_model
from .pose_estimation_models import Pose_Estimation_YOLO_Extractor, load_pose_model
from .instance_seg_models import Instance_Seg_YOLO_Extractor, load_instance_seg_model


def load_torch_model(model_type, weight, model_name):
    if model_type == "object_detection":
        return load_od_model(model_name, weight)
    elif model_type == "pose_estimation":
        return load_pose_model(model_name, weight)
    elif model_type == "instance_segmentation":
        return load_instance_seg_model(model_name, weight)
    else:
        raise "Unsupported Application"


def load_onnx_extractor(
    model_type, model_name, nc, input_name, input_shape, num_anchors
):
    if model_type == "object_detection":
        return ObjDet_YOLO_Extractor(
            model_name, nc, input_name, input_shape, num_anchors
        )
    elif model_type == "pose_estimation":
        return Pose_Estimation_YOLO_Extractor(
            model_name, nc, input_name, input_shape, num_anchors
        )
    elif model_type == "instance_segmentation":
        return Instance_Seg_YOLO_Extractor(
            model_name, nc, input_name, input_shape, num_anchors
        )
    else:
        raise "Unsupported Application"
