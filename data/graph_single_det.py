import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov9 = ["9t", "9s", "9m", "9c"]
yolov9_mAP = [35.4, 43.4, 48.6, 49.5]
yolov9_latency = [2.60, 5.36, 12.59, 14.86]

yolov8 = ["8n", "8s", "8m", "8l"]
yolov8_mAP = [34.7, 42.4, 47.6, 50.4]
yolov8_latency = [1.89, 4.17, 11.67, 24.69]

yolov5u = ["5nu", "5su", "5mu", "5lu", "5xu"]
yolov5u_mAP = [31.8, 40.4, 46.3, 49.2, 50.4]
yolov5u_latency = [1.34, 2.95, 6.49, 11.75, 22.98]

yolov5 = ["5n", "5s", "5m", "5l", "5x"]
yolov5_mAP = [26.0, 35.5, 43.0, 46.7, 48.7]
yolov5_latency = [1.26, 2.45, 5.88, 10.33, 20.49]

yolov5_6 = ["5n6"]
yolov5_6_mAP = [33.3]
yolov5_6_latency = [3.51]

# 그래프 스타일
plt.figure(figsize=(10, 6))

plt.plot(
    yolov9_latency, yolov9_mAP, marker="o", color="hotpink", linewidth=2, label="YOLOv9"
)
for i, label in enumerate(yolov9):  # 각 포인트에 라벨 추가
    plt.annotate(
        label,
        (yolov9_latency[i], yolov9_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.plot(
    yolov8_latency, yolov8_mAP, marker="o", color="blue", linewidth=2, label="YOLOv8"
)
for i, label in enumerate(yolov8):
    plt.annotate(
        label,
        (yolov8_latency[i], yolov8_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.plot(
    yolov5u_latency,
    yolov5u_mAP,
    marker="o",
    color="green",
    linewidth=2,
    label="YOLOv5u",
)
for i, label in enumerate(yolov5u):
    plt.annotate(
        label,
        (yolov5u_latency[i], yolov5u_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.plot(
    yolov5_latency, yolov5_mAP, marker="o", color="red", linewidth=2, label="YOLOv5"
)
for i, label in enumerate(yolov5):
    plt.annotate(
        label,
        (yolov5_latency[i], yolov5_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.plot(
    yolov5_6_latency,
    yolov5_6_mAP,
    marker="o",
    color="cyan",
    linewidth=2,
    label="YOLOv5-6",
)
for i, label in enumerate(yolov5_6):
    plt.annotate(
        label,
        (yolov5_6_latency[i], yolov5_6_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# 축 및 제목 설정
plt.xlabel("Latency on Warboy Single PE (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Object Detection Models' Performance on Warboy Single PE")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("data/images/graph_single_det.png", dpi=300)
plt.show()
