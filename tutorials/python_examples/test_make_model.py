from ...src.warboy.tools.onnx_tools import OnnxTools

cfg = "../src/warboy/cfg/object_detecion/yolov8n.yaml"
onnx_tools = OnnxTools(cfg)

onnx_tools.export_onnx(need_edit=False)
onnx_tools.quantize()
