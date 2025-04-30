from typing import Any, Dict

import onnx


def get_onnx_graph_info(
    task: str, model_name: str, onnx_path: str, edit_info: Dict[str, Any] = None
):
    import onnxruntime

    onnx_session = onnxruntime.InferenceSession(onnx_path)
    input_tensor = onnx_session.get_inputs()
    output_tensor = onnx_session.get_outputs()

    if len(input_tensor) != 1:
        raise "Input of YOLO must be a tensor for Image"

    input_tensor = input_tensor[0]

    input_to_shape = {
        input_tensor.name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
            for dimension_size in input_tensor.shape
        ]
    }
    if edit_info is None:
        get_output_to_shape = dict(
            object_detection=_get_output_to_shape_det,
            pose_estimation=_get_output_to_shape_pose,
            instance_segmentation=_get_output_to_shape_seg,
        )
        output_to_shape = {
            tensor_name: [
                onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
                for dimension_size in tensor_shape
            ]
            for tensor_name, tensor_shape in get_output_to_shape[task](
                model_name, input_tensor.shape, output_tensor
            )
        }
    else:
        output_to_shape = {
            tensor_name: [
                onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
                for dimension_size in edit_info[tensor_name]
            ]
            for tensor_name in edit_info
        }

    return (input_to_shape, output_to_shape)


def _get_output_to_shape_det(model_name, input_shape, output_tensor):
    import re

    output_to_shape = []
    bs, c, h, w = input_shape
    nc = 0
    num_anchors = 3

    if "yolov8" in model_name or "yolov9" in model_name:
        model_layer = {"yolov9e": "/model.42"}
        if model_name in model_layer:
            box_tensor = model_layer[model_name] + "/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = model_layer[model_name] + "/cv3.%d/cv3.%d.2/Conv_output_0"
        else:
            box_tensor = "/model.22/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = "/model.22/cv3.%d/cv3.%d.2/Conv_output_0"
        num_anchors = len(output_tensor) - 1
        nc = output_tensor[0].shape[1] - 4

        for idx in range(num_anchors):
            h_tensor = int(h / (8 * (1 << idx)))
            w_tensor = int(w / (8 * (1 << idx)))
            box_layer = (box_tensor % (idx, idx), (bs, 64, h_tensor, w_tensor))
            cls_layer = (cls_tensor % (idx, idx), (bs, nc, h_tensor, w_tensor))
            output_to_shape.append(box_layer)
            output_to_shape.append(cls_layer)
    elif re.search(r"yolov5.*u", model_name):
        if "6u" in model_name:
            box_tensor = "/model.33/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = "/model.33/cv3.%d/cv3.%d.2/Conv_output_0"
        else:
            box_tensor = "/model.24/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = "/model.24/cv3.%d/cv3.%d.2/Conv_output_0"
        num_anchors = len(output_tensor) - 1
        nc = output_tensor[0].shape[1] - 4

        for idx in range(num_anchors):
            h_tensor = int(h / (8 * (1 << idx)))
            w_tensor = int(w / (8 * (1 << idx)))
            box_layer = (box_tensor % (idx, idx), (bs, 64, h_tensor, w_tensor))
            cls_layer = (cls_tensor % (idx, idx), (bs, nc, h_tensor, w_tensor))
            output_to_shape.append(box_layer)
            output_to_shape.append(cls_layer)
    elif "yolov7" in model_name or "yolov5" in model_name:
        nc = output_tensor[0].shape[-1] - 5
        num_anchors = (
            3
            if (output_tensor[0].shape[1] // 3)
            == sum(
                [
                    int(h / (8 * (1 << idx))) * int(w / (8 * (1 << idx)))
                    for idx in range(3)
                ]
            )
            else 4
        )
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
            "yolov5x6": "/model/model/model.33",
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
                (1, 3 * (nc + 5), int(h / (8 * (1 << idx))), int(w / (8 * (1 << idx)))),
            )
            for idx in range(num_anchors)
        ]
    else:
        raise "Unsupported Object Detection Model!!"
    return output_to_shape


def _get_output_to_shape_pose(model_name, input_shape, output_tensor, num_keypoints=17):
    output_to_shape = []
    bs, c, h, w = input_shape
    nc = 0
    num_anchors = 3
    if "yolov8" in model_name:
        nc = output_tensor[0].shape[1] - num_keypoints * 3 - 4
        box_tensor = "/model.22/cv2.%d/cv2.%d.2/Conv_output_0"
        cls_tensor = "/model.22/cv3.%d/cv3.%d.2/Conv_output_0"
        skeleton_tensor = "/model.22/cv4.%d/cv4.%d.2/Conv_output_0"
        for idx in range(num_anchors):
            h_tensor = int(h / (8 * (1 << idx)))
            w_tensor = int(w / (8 * (1 << idx)))

            box_layer = (box_tensor % (idx, idx), (bs, 64, h_tensor, w_tensor))
            cls_layer = (cls_tensor % (idx, idx), (bs, nc, h_tensor, w_tensor))
            skeleton_layer = (
                skeleton_tensor % (idx, idx),
                (bs, num_keypoints * 3, h_tensor, w_tensor),
            )
            output_to_shape.append(box_layer)
            output_to_shape.append(cls_layer)
            output_to_shape.append(skeleton_layer)
    elif "yolov7" in model_name or "yolov5" in model_name:
        info = {"yolov7-w6": "/model.118", "yolov5s6": "/model.33"}
        for idx in range(num_anchors):
            box_layer_name = (
                f"{info[model_name]}/im.{idx}/Mul_output_0"
                if "yolov7" in model_name
                else f"{info[model_name]}/m.{idx}/Conv_output_0"
            )
            pose_layer_name = (
                f"{info[model_name]}/m_kpt.{idx}/m_kpt.{idx}.11/Conv_output_0"
            )
            h_tensor = int(h / (8 * (1 << idx)))
            w_tensor = int(w / (8 * (1 << idx)))

            box_layer = (box_layer_name, (bs, 6 * 3, h_tensor, w_tensor))

            pose_layer = (
                pose_layer_name,
                (bs, num_keypoints * 3 * 3, h_tensor, w_tensor),
            )
            output_to_shape.append(box_layer)
            output_to_shape.append(pose_layer)
    else:
        raise "Unsupported Pose Estimation Model!!"

    return output_to_shape


def _get_output_to_shape_seg(model_name, input_shape, output_tensor):
    output_to_shape = []
    bs, c, h, w = input_shape
    nc = output_tensor[0].shape[1] - 36
    num_anchors = 3

    if "yolov8" in model_name or "yolov9" in model_name:
        proto_layer_names = output_tensor[-1].name
        model_layer = {"yolov9e-seg": "/model.42"}
        if model_name in model_layer:
            box_tensor = model_layer[model_name] + "/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = model_layer[model_name] + "/cv3.%d/cv3.%d.2/Conv_output_0"
            instance_tensor = model_layer[model_name] + "/cv4.%d/cv4.%d.2/Conv_output_0"
        else:
            box_tensor = "/model.22/cv2.%d/cv2.%d.2/Conv_output_0"
            cls_tensor = "/model.22/cv3.%d/cv3.%d.2/Conv_output_0"
            instance_tensor = "/model.22/cv4.%d/cv4.%d.2/Conv_output_0"

        for idx in range(num_anchors):
            h_tensor = int(h / (8 * (1 << idx)))
            w_tensor = int(w / (8 * (1 << idx)))

            box_layer = (box_tensor % (idx, idx), (bs, 64, h_tensor, w_tensor))
            cls_layer = (cls_tensor % (idx, idx), (bs, nc, h_tensor, w_tensor))
            instance_layer = (
                instance_tensor % (idx, idx),
                (bs, 32, h_tensor, w_tensor),
            )

            output_to_shape.append(box_layer)
            output_to_shape.append(cls_layer)
            output_to_shape.append(instance_layer)

        proto_layer = (proto_layer_names, (bs, 32, int(h / 8) * 2, int(w / 8) * 2))
        output_to_shape.append(proto_layer)
    else:
        raise "Unsupported Segmentation Model!!"
    return output_to_shape
