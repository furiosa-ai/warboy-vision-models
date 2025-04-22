from tests.e2e import test_npu_performance

cfg = "warboy/cfg/model_config/pose_estimation/yolov8n-pose.yaml"
test_npu_performance.test_warboy_performance(cfg, 1)
