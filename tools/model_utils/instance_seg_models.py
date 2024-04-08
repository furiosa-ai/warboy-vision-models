import torch
from ultralytics import YOLO

from tools.model_utils import YOLO_ONNX_Extractor


def load_instance_seg_model(model_name, weight):
    if "yolov8" in model_name:
        return YOLO(weight).model
    else:
        raise "Unsupported Model!!"


class Instance_Seg_YOLO_Extractor(YOLO_ONNX_Extractor):
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
            proto_layer_names = {"yolov8n": "755", "yolov8m": "961", "yolov8x": "1167"}
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
                        nc,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                instance_layer = (
                    f"/model.22/cv4.{idx}/cv4.{idx}.2/Conv_output_0",
                    (
                        1,
                        32,
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                output_to_shape.append(box_layer)
                output_to_shape.append(cls_layer)
                output_to_shape.append(instance_layer)
            proto_layer = (
                proto_layer_names[model_name],
                (1, 32, int(input_shape[2] / 8) * 2, int(input_shape[3] / 8) * 2),
            )
            output_to_shape.append(proto_layer)
        else:
            raise "Unsupported Pose Estimation Model!!"

        return output_to_shape
