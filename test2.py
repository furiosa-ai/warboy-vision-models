import numpy as np
from furiosa.runtime.sync import create_runner
from ultralytics import YOLO

model = YOLO("yolov6n.pt").model
