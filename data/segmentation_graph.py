import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov9_seg = ["9c-seg", "9e-seg"]
yolov9_seg_mAP = [49.5, np.nan]
yolov9_seg_latency = [2.13, 3.78]

yolov8_seg = ["8n-seg", "8s-seg", "8m-seg", "8l-seg", "8x-seg"]
yolov8_seg_mAP = [33.9, 42.2, 47.3, 49.1, 50.4]
yolov8_seg_latency = [9.21, 9.75, np.nan, 1.51, 2.97]

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
plt.xlabel("Latency on Warboy Fusion (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Instance Segmentation Models' Performance on Warboy Fusion")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("data/images/performance_segmentation.png", dpi=300)
plt.show()
