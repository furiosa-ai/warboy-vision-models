import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov9 = ["9t", "9s", "9m", "9c"]
yolov9_mAP = [35.4, 43.4, 48.6, 49.5]
yolov9_latency = [2.13, 3.78, 9.21, 9.75]

yolov8 = ["8n", "8s", "8m", "8l", "8x"]
yolov8_mAP = [34.7, 42.4, 47.6, 50.4, 51.4]
yolov8_latency = [1.51, 2.97, 8.22, 15.22, 26.81]

yolov7 = ["7", "7x"]
yolov7_mAP = [47.9, 49.7]
yolov7_latency = [8.76, 15.61]

yolov5u = ["5nu", "5su", "5mu", "5lu", "5xu"]
yolov5u_mAP = [31.8, 40.4, 46.3, 49.2, 50.4]
yolov5u_latency = [1.15, 2.14, 4.77, 8.89, 16.51]

yolov5_6u = ["5n6u", "5s6u", "5m6u", "5l6u"]
yolov5_6u_mAP = [39.1, 45.5, 50.0, 51.8]
yolov5_6u_latency = [2.67, 5.86, 13.76, 41.85]

yolov5 = ["5n", "5s", "5m", "5l", "5x"]
yolov5_mAP = [26.0, 35.5, 43.0, 46.7, 48.7]
yolov5_latency = [1.09, 1.93, 4.44, 7.70, 14.39]

yolov5_6 = ["5n6", "5s6", "5m6", "5l6"]
yolov5_6_mAP = [33.3, 41.9, 48.7, 50.9]
yolov5_6_latency = [2.64, 5.43, 12.73, 39.17]

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
    yolov7_latency, yolov7_mAP, marker="o", color="orange", linewidth=2, label="YOLOv7"
)
for i, label in enumerate(yolov7):
    plt.annotate(
        label,
        (yolov7_latency[i], yolov7_mAP[i]),
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
    yolov5_6u_latency,
    yolov5_6u_mAP,
    marker="o",
    color="purple",
    linewidth=2,
    label="YOLOv5-6u",
)
for i, label in enumerate(yolov5_6u):
    plt.annotate(
        label,
        (yolov5_6u_latency[i], yolov5_6u_mAP[i]),
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
plt.xlabel("Latency on Warboy Fusion (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Object Detection Models' Performance on Warboy Fusion")

plt.xticks(np.arange(0, 46, 5))
plt.yticks(np.arange(25, 55, 5))

plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data/images/graph_fusion_det.png", dpi=300)
plt.show()
