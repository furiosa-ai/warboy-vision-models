import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov8_pose = ["8n-pose", "8s-pose", "8m-pose", "8l-pose"]
yolov8_pose_mAP = [47.6, 56.7, 62.2, 65.0]
yolov8_pose_latency = [1.90, 4.14, 11.9, 24.89]

# 그래프 스타일
plt.figure(figsize=(10, 6))

plt.plot(
    yolov8_pose_latency,
    yolov8_pose_mAP,
    marker="o",
    color="brown",
    linewidth=2,
    label="YOLOv8 Pose",
)
for i, label in enumerate(yolov8_pose):
    plt.annotate(
        label,
        (yolov8_pose_latency[i], yolov8_pose_mAP[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# 축 및 제목 설정
plt.xlabel("Latency on Warboy Single PE (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Pose Estimation Performance on Warboy Single PE")

plt.xticks(np.arange(0, 31, 5))
plt.yticks(np.arange(45, 68, 2.5))

plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data/images/graph_single_pose.png", dpi=300)
plt.show()
