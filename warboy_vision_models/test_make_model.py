from warboy.tools.onnx_tools import OnnxTools

# cfg = "./warboy_vision_models/warboy/cfg/model_config/object_detection/yolov8n.yaml"
# cfg = "./warboy_vision_models/warboy/cfg/model_config/pose_estimation/yolov8s-pose.yaml"
# cfg = "./warboy_vision_models/warboy/cfg/model_config/face_recognition/facenet.yaml"
cfg = "test_config.yaml"
onnx_tools = OnnxTools(cfg)

onnx_tools.export_onnx(need_edit=False)
onnx_tools.quantize()
