import onnx
from onnx.utils import Extractor


class YOLO_ONNX_Extractor:
    def __init__(self, model_name, input_name, input_shape, output_to_shape) -> None:
        self.model_name = model_name
        self.input_name = input_name
        self.input_to_shape = {
            tensor_name: [
                onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
                for dimension_size in shape
            ]
            for tensor_name, shape in [(input_name, input_shape)]
        }
        self.output_to_shape = {
            tensor_name: [
                onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
                for dimension_size in shape
            ]
            for tensor_name, shape in output_to_shape
        }

    def __repr__(self) -> str:
        return self.model_name

    def __call__(self, model):
        extracted_model = Extractor(model).extract_model(
            input_names=list(self.input_to_shape),
            output_names=list(self.output_to_shape),
        )

        for value_info in extracted_model.graph.input:
            del value_info.type.tensor_type.shape.dim[:]
            value_info.type.tensor_type.shape.dim.extend(
                self.input_to_shape[value_info.name]
            )
        for value_info in extracted_model.graph.output:
            del value_info.type.tensor_type.shape.dim[:]
            value_info.type.tensor_type.shape.dim.extend(
                self.output_to_shape[value_info.name]
            )

        return extracted_model
