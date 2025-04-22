WEIGHT_DIR = "./models/weight"
ONNX_DIR = "./models/onnx"
QUANTIZED_ONNX_DIR = "./models/quantized_onnx"
TRACE_FILE_DIR = "./models/trace"

TEST_TASK = ["object_detection", "pose_estimation", "instance_segmentation", "face_recognition"]
TEST_MODEL_LIST = {
    "object_detection": [
        "yolov5n",
        "yolov5s",
        "yolov5m",
        "yolov5l",
        "yolov5x",
        "yolov5n6",
        "yolov5s6",
        "yolov5m6",
        "yolov5l6",
        "yolov5x6",
        "yolov5nu",
        "yolov5su",
        "yolov5mu",
        "yolov5lu",
        "yolov5xu",
        "yolov5n6u",
        "yolov5s6u",
        "yolov5m6u",
        "yolov5l6u",
        "yolov5x6u",
        "yolov7",
        "yolov7x",
        #"yolov7-w6",
        #"yolov7-e6",
        #"yolov7-d6",
        #"yolov7-e6e",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolov9t",
        "yolov9s",
        "yolov9m",
        "yolov9c",
        "yolov9e",
    ],
    "pose_estimation": [
        "yolov8n-pose",
        "yolov8s-pose",
        "yolov8m-pose",
        "yolov8l-pose",
        "yolov8x-pose",
    ],
    "instance_segmentation": [
        "yolov8n-seg",
        "yolov8s-seg",
        "yolov8m-seg",
        "yolov8l-seg",
        "yolov8x-seg",
        "yolov9c-seg",
        "yolov9e-seg",
    ],
    "face_detection": [
        "yolov8n-face",
    ],
    "face_recognition": [
        "facenet",
    ],
}
