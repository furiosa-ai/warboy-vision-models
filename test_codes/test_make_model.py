from warboy.tools.onnx_tools import OnnxTools

cfg = "test_config.yaml"
onnx_tools = OnnxTools(cfg)

onnx_tools.export_onnx(need_edit=False)
onnx_tools.quantize()
