from tests.e2e import (
    test_face_recognition,
    test_instance_seg,
    test_object_det,
    test_pose_est,
)
from warboy import get_model_params_from_cfg

cfg = "test_config.yaml"

param = get_model_params_from_cfg(cfg)

if param["task"] == "face_recognition":
    test_face_recognition.test_warboy_face_recognition_accuracy(
        param["model_name"],
        param["onnx_i8_path"],
        param["input_shape"],
        param["anchors"],
    )
elif param["task"] == "instance_segmentation":
    test_instance_seg.test_warboy_instance_segmentation_accuracy(
        param["model_name"],
        param["onnx_i8_path"],
        param["input_shape"],
        param["anchors"],
    )
elif param["task"] == "object_detection":
    test_object_det.test_warboy_yolo_accuracy_det(
        param["model_name"],
        param["onnx_i8_path"],
        param["input_shape"],
        param["anchors"],
    )
elif param["task"] == "pose_estimation":
    test_pose_est.test_warboy_pose_estimation_accuracy(
        param["model_name"],
        param["onnx_i8_path"],
        param["input_shape"],
        param["anchors"],
    )
