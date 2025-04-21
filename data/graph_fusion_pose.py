import matplotlib.pyplot as plt
import numpy as np

# yolo 모델 버전별 데이터
yolov8_pose = ["8n-pose", "8s-pose", "8m-pose", "8l-pose", "8x-pose"]
yolov8_pose_mAP = [47.6, 56.7, 62.2, 65.0, 66.6]
yolov8_pose_latency = [1.54, 3.06, 8.36, 15.31, 27.55]

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
plt.xlabel("Latency on Warboy Fusion (ms/img)")
plt.ylabel("COCO mAP$_{50-95}$ val (INT8)")
plt.title("Pose Estimation Performance on Warboy Fusion")

plt.xticks(np.arange(0, 31, 5))
plt.yticks(np.arange(45, 68, 2.5))

plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data/images/graph_fusion_pose.png", dpi=300)
plt.show()
