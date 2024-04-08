# Warboy-Vision-Models
The `warboy-vision-models` project is designed to assist users in running various deep learning vision models on [FuriosaAI](https://furiosa.ai/)’s first generation NPU (Neural Processing Unit), Warboy. 
Users can follow the outlined steps in the project to execute various vision applications, such as Object Detection, Pose Estimation, Instance Segmentation, etc., alongside Warboy.

We hope that the resources here will help you utilize the FuriosaAI Warboy in your applications.

# <div align="center">Model List</div>

Currently, the project supports all vision applications provided by YOLO series ([YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv5](https://github.com/ultralytics/yolov5)). If you want to explore all available models in the `warboy-vision-models` repository and detailed performance on Warboy, please refer to the following:

### Object Detection
Object detection is a computer vision technique used to identify the presence of specific objects in images or videos and determines their locations. It entails classifying objects in videos or photos (classification) and precisely locating them using bounding boxes, thereby detecting objects through this process.

<div align="center"><img width="1024" height="360" src="./data/images/object_detection.png"></div>

<details><summary>Performance on Warboy</summary>

- [YOLOv8](https://github.com/ultralytics/ultralytics) Object Detection (COCO)

<div align="center">

| Model     | Input Size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Accuracy Drop<br>(%)    | Warboy Speed<sup>Fusion<br>(ms)     | Warboy Speed<sup>Single PE<br>(ms)     |
| --------- | --------------------------- | -------------------- |------------------------ | ----------------------------------- | -------------------------------------- |
| YOLOv8n   | 640x640                     | 34.9                 |                         |                                     |                                        |
| YOLOv8s   | 640x640                     | 42.9                 |                         |                                     |                                        |
| YOLOv8m   | 640x640                     | 47.8                 |                         |                                     |                                        |
| YOLOv8l   | 640x640                     | 50.4                 |                         |                                     |                                        |
| YOLOv8x   | 640x640                     | 51.4                 |                         |                                     |                                        |
</div>

- [YOLOv7](https://github.com/WongKinYiu/yolov7) Object Detection (COCO)
<div align="center">

| Model     | Input Size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Accuracy Drop<br>(%)    | Warboy Speed<sup>Fusion<br>(ms)     | Warboy Speed<sup>Single PE<br>(ms)     |
| --------- | --------------------------- | -------------------- |------------------------ | ----------------------------------- | -------------------------------------- |
| YOLOv7    | 640x640                     |                      |                         |                                     |                                        |
| YOLOv7x   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv7-w6 | 1280x1280                   |                      |                         |                                     |                                        |
| YOLOv7-e6 | 1280x1280                   |                      |                         |                                     |                                        |
| YOLOv7-d6 | 1280x1280                   |                      |                         |                                     |                                        |
</div>

- [YOLOv5](https://github.com/ultralytics/yolov5) Object Detection (COCO)
<div align="center">

| Model     | Input Size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Accuracy Drop<br>(%)    | Warboy Speed<sup>Fusion<br>(ms)     | Warboy Speed<sup>Single PE<br>(ms)     |
| --------- | --------------------------- | -------------------- |------------------------ | ----------------------------------- | -------------------------------------- |
| YOLOv5n   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv5s   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv5m   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv5l   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv5x   | 640x640                     |                      |                         |                                     |                                        |
| YOLOv5n6  | 1280x1280                   |                      |                         |                                     |                                        |
| YOLOv5s6  | 1280x1280                   |                      |                         |                                     |                                        |
| YOLOv5m6  | 1280x1280                   |                      |                         |                                     |                                        |
| YOLOv5l6  | 1280x1280                   |                      |                         |                                     |                                        |
</div>


</details>

### Pose Estimation
Pose estimation is a technology that identifies and estimates the posture of a person or object by detecting body parts (typically joints) and using them to estimate the pose of the respective object.

<div align="center"><img width="720" src="./data/images/pose_estimation.png"></div>

<details><summary>Performance on Warboy</summary>

</details>

### Instance Segmentation 
Instance segmentation is a technology that identifies multiple objects in an image or video and delineates the boundaries of each object. In essence, it combines Object Detection and Semantic Segmentation techniques to individually identify multiple objects belonging to the same class and estimate their boundaries.

<div align="center"><img width="720" src="./data/images/instance_segmentation.png"></div>

<details><summary>Performance on Warboy</summary>


</details>

# <div align="center">Documentation</div>
Please refer below for a installation and usage example. 

## Installation
To use this project, it's essential to install various software components provided by FuriosaAI. For detailed instructions on installing packages, drivers, and the Furiosa SDK, please see the following:

- **Driver, Firmeware and Runtime Installation** ([English](https://furiosa-ai.github.io/docs/latest/en/software/installation.html) | [한국어](https://furiosa-ai.github.io/docs/latest/ko/software/installation.html))


Python SDK requires Python 3.8 or above. pip install required python packages as follows,
```sh
pip install -r requirements.txt
```
and apt install required packages for post processing utilites.
```sh
sudo apt-get update
sudo apt-get install cmake libeigen3-dev
./build.sh
```

## Usage Example

<details open>
<summary> Set config files for project </summary>
  
First, download the weight file from [YOLOv8](https://github.com/ultralytics/ultralytics) for example execution.
  
```sh
cd warboy-vision-models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
```    

Before running the project, you need to set up configuration files for model and demo. 

- **Model config file** : it contains parameters about the model and quantization. 
```yaml
application: object_detection            # vision task (object detection | pose estimation | instance segmentation)
model_name: yolov8n                      # model name
weight: yolov8n.pt                       # weight file path
onnx_path: yolov8n.onnx                  # onnx model path
onnx_i8_path: yolov8n_i8.onnx            # quantized onnx model path

calibration_params:
  calibration_method: SQNR_ASYM               # calibration method
  calibration_data: calibration_data          # calibration data path
  num_calibration_data: 10                    # number of calibration data

confidence_threshold: 0.25
iou_threshold: 0.7
input_shape: [640, 640]         # model input shape (Height, Width)
anchors:                        # anchor information
  - 
class_names:                    # class names
  - ...
```

- **Demo config file** : it contains device informations and video paths for the project.

```yaml
application: object_detection
model_config: ./cfg/object_detection_model.yaml
model_path: yolov8n_i8.onnx
output_path: output_detection
num_workers: 8
device: warboy(2)*1
video_path: 
  - [set your test video file path]
```

</details>

<details open>
<summary> Export ONNX </summary>
Next, it is necessary to export the model to the ONNX format. 

- **command**
  ```sh
  python tools/export_onnx.py cfg/object_detection_model.yaml
  ```

</details>

<details open>
<summary> Quantizing an ONNX model using Furiosa SDK </summary>
If you have already exported the model from its original format to an ONNX model, the next step is the model quantization process. Since FuriosaAI's Warboy only supports models in 8-bit integer format (int8), it is necessary to quantize the float32-based model into an int8 data type model. 

- **command**
  ```sh
  python tools/furiosa_quantizer.py cfg/object_detection_model.yaml
  ```
</details>

<details open>
<summary> Running the project using Furiosa Runtime </summary>
In the project, vision applications are executed for videos from multiple channels. To accomplish this effectively, optimization tasks such as Python parallel programming, asynchronous processing, and post-processing using C++ have been included. For a detailed understanding of the project structure, please refer to the following image:


<div align="center"><img width="960" src="./data/images/structure.png"></div>
  
- **command**
  
  ```sh
  python warboy_demo.py cfg/demo.yaml file    # save the result as image file
  python warboy_demo.py cfg/demo.yaml fastAPI # see the result on webpage using fastAPI (http://0.0.0.0:20001)
  ```
</details>

