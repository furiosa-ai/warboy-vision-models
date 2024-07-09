import onnx
import torch

from typing import Dict, Any, List
from warboy.cfg import get_model_params_from_cfg


class WARBOY_YOLO:
    """
    A class for executing YOLO models in Warboy.

    This class provides an interface for ONNX conversion and quantization based on FuriosaAI SDK.
    
    Args:
        cfg (str): The configuration of the model if loaded from a *.yaml file

    Methods: 
        export_onnx: 
        _edit_onnx:
        _load_torch_model:
        _chck_yolo_version:
        quantize:

    Raises:
        FileNotFoundError: 
        ValueError: 
    """

    def __init__(self, cfg: str, edit_info=None) -> None:
        params = get_model_params_from_cfg(cfg)
        self.task = params["task"]
        self.model_name = params["model_name"]
        self.weight = params["weight"]
        self.onnx_path = params["onnx_path"]
        self.input_shape = params["input_shape"]
        self.onnx_i8_path = params["onnx_i8_path"]
        self.calibration_method, self.calibration_data, self.num_calibration_data = params[
            "calibration_params"
        ].values()
        self.anchors = params["anchors"]
        self.edit_info = (
            self.params["edit_info"] if "edit_info" in params else edit_info
        )
        return

    def export_onnx(self, need_edit: bool = True):
        print(f"Load PyTorch Model from {self.weight}...")
        torch_model = self._load_torch_model().eval()

        print(f"Export ONNX {self.onnx_path}...")
        dummy_input = torch.zeros(*self.input_shape).to(torch.device("cpu"))

        torch.onnx.export(
            torch_model,
            dummy_input,
            self.onnx_path,
            opset_version=self.opset_version,
            input_names=["images"],
            output_names=["outputs"],
        )

        if need_edit:
            edited_model = self._edit_onnx(self.edit_info)
            onnx.save(
                onnx.save(onnx.shape_inference.infer_shapes(edited_model), onnx_path)
            )

        print(f"Export ONNX for {self.model_name} >> {self.onnx_path}")
        return

    def _edit_onnx(self, edit_info: Dict[str, List] = None):
        from onnx.utils import Extractor

        onnx_graph = onnx.load(self.onnx_path)
        input_to_shape, output_to_shape = _get_onnx_graph_info(
            self.task, onnx_graph, edit_info
        )
        edited_graph = Extractor(onnx_graph).extract_model(
            input_names=list(input_to_shape), output_names=list(output_to_shape)
        )

        for value_info in edited_graph.graph.input:
            del value_info.type.tensor_type.shape.dim[:]
            value_info.type.tensor_type.shape.dim.extend(
                input_to_shape[value_info.name]
            )
        for value_info in edited_graph.graph.output:
            del value_info.type.tensor_type.shape.dim[:]
            value_info.type.tensor_type.shape.dim.extend(
                output_to_shape[value_info.name]
            )
        return edited_graph

    def _load_torch_model(self):
        """
        """
        from ultralytics import YOLO

        if not self.model_name in MODEL_LIST[self.task]:
            raise ValueError(
                f"Supported Model List (model_name) for {self.task}:\n {','.join(MODEL_LIST[self.task])}\n"
            )

        yolo_version = self._check_yolo_version()

        if yolo_version >= 8 and yolo_version < 10:
            torch_model = YOLO(self.weight).model
        elif yolo_version == 7:
            torch_model = torch.hub.load("WongKinYiu/yolov7", "custom", self.weight)
        elif yolo_version == 5:
            torch_model = torch.hub.load("ultralytics/yolov5", "custom", self.weight)
        else:
            raise ValueError(f"Supported Version 5,7,8,9")

        return torch_model

    @property
    def _check_yolo_version(self):
        if "yolov9" in self.model_name:
            return 9
        elif "yolov8" in self.model_name:
            return 8
        elif "yolov7" in self.model_name:
            return 7
        elif "yolov5" in self.model_name:
            return 5
        else:
            return -1

    def quantize(self, use_model_editor: bool = True):
        """
        Qauntize the model using FuriosaAI SDK.
        """
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"{self.onnx_path} is not found!")

        from furiosa.optimizer import optimize_model
        from furiosa.quantizer import (
            CalibrationMethod,
            Calibrator,
            ModelEditor,
            TensorType,
            get_pure_input_names,
            quantize,
        )

        onnx_model = onnx.load(self.onnx_path)
        onnx_model = optimize_model(
            model=onnx_model,
            opset_version=self.opset_version,
            input_shapes={"images": [1, 3, *self.input_shape]},
        )

        calibrator = Calibrator(
            model, CalibrationMethod._member_map_[self.calibration_method]
        )

        if use_model_editor:
            editor = ModelEditor(onnx_model)
            input_names = get_pure_input_names(onnx_model)

            for input_name in input_names:
                editor.convert_input_type(input_name, TensorType.UINT8)

        calib_range = calibrator.compute_range()
        quantized_model = quantize(onnx_model, calib_range)

        with open(self.onnx_i8_path, "wb") as f:
            f.write(bytes(quantized_model))

        print(f"Quantization completd >> {self.onnx_i8_path}")
        return

    def _get_calibration_dataset(self):
        calibration_data = None
