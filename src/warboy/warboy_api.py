from .utils.process_pipeline import Engine, PipeLine, Video

QUEUE_SIZE = 500


def set_demo_engin_config(param, idx):
    """Set engine config for each task"""
    engin_config = {
        "name": f"test_{param['task']}_{idx}",
        "task": param["task"],
        "model": param["model_path"],
        "worker_num": param["worker_num"],
        "device": param["warboy_device"],
        "model_type": param["model_name"],
        "input_shape": param["input_shape"],
        "class_names": param["class_name"],
        "iou_thres": param["model_param"].get("iou_thres", 0.7),
        "conf_thres": param["model_param"].get("conf_thres", 0.25),
    }
    return engin_config


class AppRunner:
    """ """

    def __init__(self, params, demo_type) -> None:
        num_videos = sum(len(param["videos_info"]) for param in params)

        # Warboy Runtime
        if demo_type == "web":
            self.job_handler = PipeLine(num_channels=num_videos)
        elif demo_type == "image":
            self.job_handler = PipeLine(
                num_channels=num_videos, run_fast_api=False, make_image_output=True
            )
        elif demo_type == "file":
            self.job_handler = PipeLine(
                num_channels=num_videos, run_fast_api=False, make_file_output=True
            )

        engin_configs_dict = {}
        videos = {}
        for param in params:
            task = param["task"]
            if not task in engin_configs_dict:
                engin_configs_dict[task] = []
                videos[task] = []

            engin_configs_dict[task].append(
                set_demo_engin_config(param, len(engin_configs_dict[task]))
            )

            videos[task] += [
                Video(
                    video_info=video_info["input_path"],
                    recursive=video_info["recursive"],
                )
                for video_info in param["videos_info"]
            ]

        for task, engin_configs in engin_configs_dict.items():
            for idx, engin in enumerate(engin_configs):
                if demo_type == "file":
                    self.job_handler.add(
                        Engine(**engin),
                        postprocess_as_img=False,
                    )
                else:
                    self.job_handler.add(Engine(**engin))

                for video_idx in range(idx, len(videos[task]), len(engin_configs)):
                    self.job_handler.add(videos[task][video_idx], name=engin["name"])

    def __call__(self):
        self.job_handler.run()
        return


class WARBOY_APP:
    """ """

    def __init__(self, params, demo_type):
        self.params = params
        self.demo_type = demo_type

        self.result_queue = []

        self.app_runner = AppRunner(self.params, self.demo_type)

    def get_result_queues(self):
        return self.app_runner.job_handler.result_mux_list

    def __call__(self):
        self.app_runner()
