from typing import Optional

import numpy as np
import typer

_ = np.finfo(np.float64)
_ = np.finfo(np.float32)

from demo.demo import run_make_file, run_web_demo
from test_scenarios.e2e import (
    test_face_recognition,
    test_instance_seg,
    test_npu_performance,
    test_object_det,
    test_pose_est,
)
from warboy import get_model_params_from_cfg
from warboy.tools.onnx_tools import OnnxTools

app = typer.Typer()


@app.command("face-recognition", help="Run end-to-end test for face recognition.")
def face_recognition_e2e_tests(
    model_name: str,
    onnx_i8: str,
):
    cfg = f"tests/test_config/face_recognition/{model_name}.yaml"
    param = get_model_params_from_cfg(cfg)

    test_face_recognition.test_warboy_facenet_accuracy_recog(
        model_name, onnx_i8, param["input_shape"], param["anchors"],
        "datasets/face_recognition/lfw-align-128",
        "datasets/face_recognition/lfw_test_pair.txt",
    )


@app.command(
    "instance-segmentation", help="Run end-to-end test for instance segmentation."
)
def instance_segmentation_e2e_tests(
    model_name: str,
    onnx_i8: str,
):
    cfg = f"tests/test_config/instance_segmentation/{model_name}.yaml"
    param = get_model_params_from_cfg(cfg)

    test_instance_seg.test_warboy_yolo_accuracy_seg(
        model_name, onnx_i8, param["input_shape"], param["anchors"],
        "datasets/coco/val2017",
        "datasets/coco/annotations/instances_val2017.json",
    )


@app.command("object-detection", help="Run end-to-end test for object detection.")
def object_detection_e2e_test(
    model_name: str,
    onnx_i8: str,
):
    cfg = f"tests/test_config/object_detection/{model_name}.yaml"
    param = get_model_params_from_cfg(cfg)

    test_object_det.test_warboy_yolo_accuracy_det(
        model_name, onnx_i8, param["input_shape"], param["anchors"],
        "datasets/coco/val2017",
        "datasets/coco/annotations/instances_val2017.json",
    )


@app.command("pose-estimation", help="Run end-to-end test for pose estimation.")
def pose_estimation_e2e_test(
    model_name: str,
    onnx_i8: str,
):
    cfg = f"tests/test_config/pose_estimation/{model_name}.yaml"
    param = get_model_params_from_cfg(cfg)

    test_pose_est.test_warboy_yolo_accuracy_pose(
        model_name, onnx_i8, param["input_shape"], param["anchors"],
        "datasets/coco/val2017",
        "datasets/coco/annotations/person_keypoints_val2017.json",
    )


@app.command("performance", help="Run end-to-end test with config file.")
def run_e2e_tests(
    cfg: str,
):
    param = get_model_params_from_cfg(cfg)
    func = None
    if param["task"] == "face_recognition":
        func = test_face_recognition.test_warboy_facenet_accuracy_recog
        dataset = "datasets/face_recognition/lfw-align-128"
        annotation = "datasets/face_recognition/lfw_test_pair.txt"
    elif param["task"] == "instance_segmentation":
        func = test_instance_seg.test_warboy_yolo_accuracy_seg
        dataset = "datasets/coco/val2017"
        annotation = "datasets/coco/annotations/instances_val2017.json"
    elif param["task"] == "object_detection":
        func = test_object_det.test_warboy_yolo_accuracy_det
        dataset = "datasets/coco/val2017"
        annotation = "datasets/coco/annotations/instances_val2017.json"
    elif param["task"] == "pose_estimation":
        func = test_pose_est.test_warboy_yolo_accuracy_pose
        dataset = "datasets/coco/val2017"
        annotation = "datasets/coco/annotations/person_keypoints_val2017.json"
    else:
        typer.echo(f"Error: Unsupported task '{param['task']}' in the config file.")

    func(
        param["model_name"],
        param["onnx_i8_path"],
        param["input_shape"],
        param["anchors"],
        dataset,
        annotation,
    )


@app.command("npu-profile", help="Run NPU performance test.")
def npu_performance_test(
    cfg_path: str,
    num_device: Optional[int] = 1,
):
    test_npu_performance.test_warboy_performance(cfg_path, num_device)


@app.command("web-demo", help="Run web demo.")
def web_demo(
    cfg_path,
):
    run_web_demo(cfg_path)

@app.command("make-file", help="Run file demo.")
def make_file_demo(
    cfg_path,
):
    run_make_file(cfg_path)

@app.command("make-model", help="Export model to ONNX format and quantize it.")
def make_model(
    cfg: str,
    need_edit: Optional[bool] = True,
    quantize: Optional[bool] = True,
):
    onnx_tools = OnnxTools(cfg)

    if need_edit and "yolo" not in onnx_tools.model_name:
        typer.echo(
            "Warning: The model is not a YOLO model. The need_edit option is ignored."
        )
        need_edit = False
    onnx_tools.export_onnx(need_edit=need_edit)
    if quantize:
        onnx_tools.quantize()


@app.command("export-onnx", help="Export model to ONNX format.")
def export_onnx(
    cfg: str,
    need_edit: Optional[bool] = True,
):
    onnx_tools = OnnxTools(cfg)

    if need_edit and "yolo" not in onnx_tools.model_name:
        typer.echo(
            "Warning: The model is not a YOLO model. The need_edit option is ignored."
        )
        need_edit = False
    onnx_tools.export_onnx(need_edit=need_edit)


@app.command("quantize", help="Quantize the ONNX model.")
def quantize(
    cfg: str,
):
    onnx_tools = OnnxTools(cfg)
    onnx_tools.quantize()


if __name__ == "__main__":
    app()
