# Warboy-Vsion-Models
`warboy-vision-models` is a project, designed to assist users execute various deep learning vision models on [FuriosaAI](https://furiosa.ai/)’s first generation NPU(Neural Processing Unit) Warboy. 
Users can follow the steps outlined in the project to execute various vision applications, such as Object Detection, Pose Estimation, Instance Segmentation, etc., alongside Warboy.

We hope that the resources here will help you use the FuriosaAI Warboy on your applications.

# <div align="center">Model List</div>

Currently, the project supports all vision applications provided by YOLO series ([YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv5](https://github.com/ultralytics/yolov5)). If you want to know all models available in `warboy-vision-models` and detailed performance of Warboy, please see the followings:

### Object Detection
Object detection is a computer vision technique that identifies the presence of specific objects in images or videos and determines their locations. It involves classifying objects in videos or photos (classification) and pinpointing their locations through bounding boxes, thus detecting objects through this process.

<img width="1024" src=>

<details><summary>Performance Table</summary>
  
| Model   | Size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>Warboy Fusion<br>(ms) | Speed<br><sup>Warboy Single PE<br>(ms) |
| ------- | --------------------- | -------------------- | ----------------------------------- | -------------------------------------- |
| YOLOv8n | 640                   | 34.9                 |                                     |                                        |


</details>

### Pose Estimation

<details><summary>Performance Table</summary>

</details>

### Instance Segmentation 

<details><summary>Performance Table</summary>


</details>

# <div align="center">Documentation</div>
[설명 넣기]

## Installation
For the project, it is necessary to install various software componenets provided by FuriosaAI. If you want to learn more about the installation of package, driver and furiosa-sdk, please refer to the followings:

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

## Performance

## Usage Example

<details open>
<summary> Set config files for project </summary>
  
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

conf_thres: 0.25
iou_thres: 0.7
input_shape: [640, 640]         # model input shape (Height, Width)
anchors:                        # anchor information
  - 
class_names:                    # class names
  - ...
```

- **Demo config file** : 

```yaml
application: object_detection
model_config: ./cfg/model_config.yaml
model_path: yolov8n_i8.onnx
output_path: output_detection
num_worker: 8
device: warboy(2)*1
video_path: 
  - ...
```

</details>

<details open>
<summary> Export ONNX </summary>
Next, it is necessary to export the model to the ONNX format. 

- **command**
  ```sh
  python tools/export_onnx.py cfg/model_config.yaml
  ```

</details>

<details open>
<summary> Quantize ONNX using Furiosa SDK </summary>
If you have proceeded with the conversion of the model from its original format to an ONNX model, the next step is the model quantization process. Since FuriosaAI's Warboy only supports models in 8-bit integer format (int8), it is necessary to quantize the float32-based model into an int8 data type model. 

- **command**
  ```sh
  python tools/furiosa_quantizer.py cfg/model_config.yaml
  ```
</details>

<details open>
<summary> Run project using Furiosa Runtime </summary>
  
- **command**
  
  ```sh
  python warboy_demo.py cfg/demo.config.yaml file
  ```
</details>

