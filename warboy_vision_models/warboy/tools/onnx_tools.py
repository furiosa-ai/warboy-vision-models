import os
from typing import Dict, List

import cv2
import onnx
import torch
from tqdm import tqdm

from warboy_vision_models.warboy.cfg import MODEL_LIST, get_model_params_from_cfg
from warboy_vision_models.warboy.tools.utils import get_onnx_graph_info


class OnnxTools:
    def __init__(self, cfg: str):
        params = get_model_params_from_cfg(cfg)
        self.task = params["task"]
        self.model_name = params["model_name"]
        self.weight = params["weight"]
        self.onnx_path = params["onnx_path"]
        self.input_shape = params["input_shape"]
        self.onnx_i8_path = params["onnx_i8_path"]
        self.opset_version = 13
        self.calibration_method, self.calibration_data, self.num_calibration_data = (
            params["calibration_params"].values()
        )
        self.anchors = params["anchors"]
        return

    def export_onnx(self, need_edit: bool = True, edit_info=None):
        print(f"Load PyTorch Model from {self.weight}...")
        if os.path.dirname(self.onnx_path) != "" and not os.path.exists(
            os.path.dirname(self.onnx_path)
        ):
            os.makedirs(os.path.dirname(self.onnx_path))

        if "yolo" in self.model_name:
            torch_model = self._load_yolo_torch_model().eval()
        elif "facenet" == self.model_name:
            torch_model = self._load_facenet_torch_model().eval()
        else:
            raise NotImplementedError(
                f"Export ONNX for {self.task} >> {self.model_name} is not implemented yet!"
            )

        print(f"Export ONNX {self.onnx_path}...")
        dummy_input = torch.randn(*self.input_shape).to(torch.device("cpu"))

        torch.onnx.export(
            torch_model,
            dummy_input,
            self.onnx_path,
            opset_version=self.opset_version,
            input_names=["images"],
            output_names=["outputs"],
        )

        if need_edit:
            edited_model = self._edit_onnx(edit_info)
            onnx.save(onnx.shape_inference.infer_shapes(edited_model), self.onnx_path)

        print(f"Export ONNX for {self.model_name} >> {self.onnx_path}")
        return True

    def _edit_onnx(self, edit_info: Dict[str, List] = None):
        from onnx.utils import Extractor

        onnx_graph = onnx.load(self.onnx_path)
        input_to_shape, output_to_shape = get_onnx_graph_info(
            self.task, self.model_name, self.onnx_path, edit_info
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

    def _load_yolo_torch_model(self):
        """ """
        from ultralytics import YOLO

        if not self.model_name in MODEL_LIST[self.task]:
            raise ValueError(
                f"Supported Model List (model_name) for {self.task}:\n {','.join(MODEL_LIST[self.task])}\n"
            )

        yolo_version = self._check_yolo_version

        if yolo_version >= 8 and yolo_version < 10:
            if self.task == "instance_segmentation" and yolo_version == 9:
                torch_model = torch.hub.load(
                    "WongKinYiu/yolov9", "custom", self.weight
                ).to(torch.device("cpu"))
            else:
                torch_model = YOLO(self.weight).model
        elif yolo_version == 7:
            torch_model = torch.hub.load("WongKinYiu/yolov7", "custom", self.weight).to(
                torch.device("cpu")
            )
        elif yolo_version == 5 and "u" in self.model_name:
            torch_model = YOLO(self.weight).model
        elif yolo_version == 5:
            torch_model = torch.hub.load(
                "ultralytics/yolov5", "custom", self.weight
            ).to(torch.device("cpu"))
        else:
            raise ValueError(f"Supported Version 5,7,8,9")

        return torch_model

    def _load_facenet_torch_model(self):
        """
        Load Facenet Torch Model
        """
        from facenet_pytorch import InceptionResnetV1

        torch_model = InceptionResnetV1(pretrained="vggface2")

        return torch_model

    @property
    def _check_yolo_version(self):
        if "v9" in self.model_name:
            return 9
        elif "v8" in self.model_name:
            return 8
        elif "v7" in self.model_name:
            return 7
        elif "v5" in self.model_name:
            return 5
        else:
            return -1

    def quantize(self, use_model_editor: bool = True):
        """
        Qauntize the model using FuriosaAI SDK.
        """
        if not os.path.exists(os.path.dirname(self.onnx_i8_path)):
            os.makedirs(os.path.dirname(self.onnx_i8_path))

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

        new_shape = self.input_shape[2:]
        onnx_model = onnx.load(self.onnx_path)
        onnx_model = optimize_model(
            model=onnx_model,
            opset_version=self.opset_version,
            input_shapes={"images": self.input_shape},
        )

        calibrator = Calibrator(
            onnx_model, CalibrationMethod._member_map_[self.calibration_method]
        )
        if "yolo" in self.model_name:
            from warboy_vision_models.warboy.yolo.preprocess import YoloPreProcessor

            preprocessor = YoloPreProcessor(new_shape=new_shape, tensor_type="float32")

        elif "face_recognition" == self.task:
            from warboy_vision_models.warboy.face_recognition.preprocess import (
                FaceRecogPreProcessor,
            )

            preprocessor = FaceRecogPreProcessor(
                new_shape=new_shape, tensor_type="float32"
            )

        else:
            raise NotImplementedError(
                f"Preprocessor for {self.task} >> {self.model_name} is not implemented yet!"
            )

        for calibration_data in tqdm(
            self._get_calibration_dataset(), desc="calibration..."
        ):
            input_img = cv2.imread(calibration_data)
            input_, _ = preprocessor(input_img)
            calibrator.collect_data([[input_]])

        if use_model_editor:
            editor = ModelEditor(onnx_model)
            input_names = get_pure_input_names(onnx_model)

            for input_name in input_names:
                editor.convert_input_type(input_name, TensorType.UINT8)

        calib_range = calibrator.compute_range()
        quantized_model = quantize(onnx_model, calib_range)

        with open(self.onnx_i8_path, "wb") as f:
            f.write(bytes(quantized_model))

        print(f"Quantization completed >> {self.onnx_i8_path}")
        return True

    def _get_calibration_dataset(self):
        import glob
        import imghdr
        import random

        calibration_data = []

        datas = glob.glob(self.calibration_data + "/**", recursive=True)
        datas = random.choices(datas, k=min(self.num_calibration_data, len(datas)))
        print(datas)
        for data in datas:
            if os.path.isdir(data) or imghdr.what(data) is None:
                continue
            calibration_data.append(data)
        print(calibration_data)
        return calibration_data
