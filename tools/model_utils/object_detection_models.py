import torch
from ultralytics import YOLO

from tools.model_utils import YOLO_ONNX_Extractor


def load_od_model(model_name, weight):
    if "yolov8" in model_name or "yolov9" in model_name:
        return YOLO(weight).model
    elif "yolov7" in model_name:
        return torch.hub.load("WongKinYiu/yolov7", "custom", weight)
    elif "yolov5" in model_name:
        return torch.hub.load("ultralytics/yolov5", "custom", weight)
    else:
        raise "Unsupported Model!!"


class ObjDet_YOLO_Extractor(YOLO_ONNX_Extractor):
    def __init__(self, model_name, nc, input_name, input_shape, num_anchors=3):
        super().__init__(
            model_name,
            input_name,
            input_shape,
            self.get_output_to_shape(model_name, nc, input_shape, num_anchors),
        )

    def get_output_to_shape(self, model_name, nc, input_shape, num_anchors):
        output_to_shape = []
        if "yolov8" in model_name or "yolov9" in model_name:
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
                output_to_shape.append(box_layer)
                output_to_shape.append(cls_layer)
        elif "yolov7" in model_name:
            info = {
                "yolov7": "/model/model.105",
                "yolov7x": "/model/model.121",
                "yolov7-w6": "/model/model.118",
                "yolov7-e6": "/model/model.140",
                "yolov7-d6": "/model/model.162",
                "yolov7-e6e": "/model/model.261",
            }
            output_to_shape = [
                (
                    f"{info[model_name]}/m.{idx}/Conv_output_0",
                    (
                        1,
                        3 * (nc + 5),
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                for idx in range(num_anchors)
            ]
        elif "yolov5" in model_name:
            info = {
                "yolov5n": "/model/model/model.24",
                "yolov5s": "/model/model/model.24",
                "yolov5m": "/model/model/model.24",
                "yolov5l": "/model/model/model.24",
                "yolov5x": "/model/model/model.24",
                "yolov5n6": "/model/model/model.33",
                "yolov5s6": "/model/model/model.33",
                "yolov5m6": "/model/model/model.33",
                "yolov5l6": "/model/model/model.33",
            }
            output_to_shape = [
                (
                    f"{info[model_name]}/m.{idx}/Conv_output_0",
                    (
                        1,
                        3 * (nc + 5),
                        int(input_shape[2] / (8 * (1 << idx))),
                        int(input_shape[3] / (8 * (1 << idx))),
                    ),
                )
                for idx in range(num_anchors)
            ]
        else:
            raise "Unsupported Object Detection Model!!"

        return output_to_shape
