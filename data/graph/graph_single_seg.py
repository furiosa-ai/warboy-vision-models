import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov9_seg = ["9c-seg"]
yolov9_seg_mAP = [49.5]
yolov9_seg_latency = [2.60]

yolov8_seg = ["8n-seg", "8s-seg"]
yolov8_seg_mAP = [33.9, 42.2]
yolov8_seg_latency = [12.59, 14.86]

# 그래프 스타일
plt.figure(figsize=(10, 6))

plt.plot(
    yolov9_seg_latency,
    yolov9_seg_mAP,
    marker="o",
    color="pink",
    linewidth=2,
    label="YOLOv9 Seg",
)
for i, label in enumerate(yolov9_seg):
    plt.annotate(
        label,
        (yolov9_seg_latency[i], yolov9_seg_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.plot(
    yolov8_seg_latency,
    yolov8_seg_mAP,
    marker="o",
    color="gray",
    linewidth=2,
    label="YOLOv8 Seg",
)
for i, label in enumerate(yolov8_seg):
    plt.annotate(
        label,
        (yolov8_seg_latency[i], yolov8_seg_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# 축 및 제목 설정
plt.xlabel("Latency on Warboy Single PE (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Instance Segmentation Models' Performance on Warboy Single PE")

plt.xticks(np.arange(0, 17, 2))
plt.yticks(np.arange(30, 51, 2))

plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data/images/graph_single_seg.png", dpi=300)
plt.show()
