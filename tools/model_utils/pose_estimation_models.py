import torch
from ultralytics import YOLO

from tools.model_utils import YOLO_ONNX_Extractor


def load_pose_model(model_name, weight):
    if "yolov8" in model_name:
        return YOLO(weight).model
    elif "yolov7" in model_name:
        import sys, os

        sys.path.append(
            os.path.join(
                os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "yolov7"
            )
        )
        from yolov7.models.experimental import attempt_load

        model = attempt_load(weight, map_location="cpu")
        return model
    elif "yolov5" in model_name:
        import sys, os

        sys.path.append(
            os.path.join(
                os.path.dirname(os.path.abspath(os.path.dirname(__file__))),
                "edgeai_yolov5",
            )
        )
        from edgeai_yolov5.models.experimental import attempt_load

        model = attempt_load(weight, map_location="cpu")
        return model
    else:
        raise "Unsupported Model!!"


class Pose_Estimation_YOLO_Extractor(YOLO_ONNX_Extractor):
    def __init__(self, model_name, nc, input_name, input_shape, num_anchors=3):
        super().__init__(
            model_name,
            input_name,
            input_shape,
            self.get_output_to_shape(model_name, nc, input_shape, num_anchors),
        )

    def get_output_to_shape(self, model_name, nc, input_shape, num_anchors):
        output_to_shape = []
        if "yolov8" in model_name:
            for idx in range(num_anchors):
                box_layer = (
                    f"/model.22/cv2.{idx}/cv2.{idx}.2/Conv_output_0",
                    (
                        1,
                        64,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                cls_layer = (
                    f"/model.22/cv3.{idx}/cv3.{idx}.2/Conv_output_0",
                    (
                        1,
                        1,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                skeleton_layer = (
                    f"/model.22/cv4.{idx}/cv4.{idx}.2/Conv_output_0",
                    (
                        1,
                        17 * 3,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                output_to_shape.append(box_layer)
                output_to_shape.append(cls_layer)
                output_to_shape.append(skeleton_layer)
        elif "yolov7" in model_name:
            info = {"yolov7-w6": "/model.118"}
            for idx in range(num_anchors):
                box_layer_name = f"{info[model_name]}/im.{idx}/Mul_output_0"
                pose_layer_name = (
                    f"{info[model_name]}/m_kpt.{idx}/m_kpt.{idx}.11/Conv_output_0"
                )

                box_layer = (
                    box_layer_name,
                    (
                        1,
                        6 * 3,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )

                pose_layer = (
                    pose_layer_name,
                    (
                        1,
                        17 * 3 * 3,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                output_to_shape.append(box_layer)
                output_to_shape.append(pose_layer)

        elif "yolov5" in model_name:
            info = {"yolov5s6": "/model.33"}
            for idx in range(num_anchors):
                box_layer_name = f"{info[model_name]}/m.{idx}/Conv_output_0"
                pose_layer_name = (
                    f"{info[model_name]}/m_kpt.{idx}/m_kpt.{idx}.11/Conv_output_0"
                )

                box_layer = (
                    box_layer_name,
                    (
                        1,
                        6 * 3,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )

                pose_layer = (
                    pose_layer_name,
                    (
                        1,
                        17 * 3 * 3,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                output_to_shape.append(box_layer)
                output_to_shape.append(pose_layer)
        else:
            raise "Unsupported Pose Estimation Model!!"

        return output_to_shape
