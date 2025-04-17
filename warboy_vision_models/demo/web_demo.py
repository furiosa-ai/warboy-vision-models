import threading

from warboy_vision_models.warboy import (
    WARBOY_APP,
    get_demo_params_from_cfg,
    spawn_web_viewer,
)


def run_warboy_app(warboy_app):
    t = threading.Thread(target=warboy_app)
    t.daemon = True
    t.start()
    return t.native_id


def run_web_demo(cfg_path="./warboy_vision_models/warboy/cfg/demo_config/demo.yaml"):
    demo_params = get_demo_params_from_cfg(cfg_path)
    warboy_app = WARBOY_APP(demo_params)
    run_warboy_app(warboy_app)
    result_queues = warboy_app.get_result_queues()
    spawn_web_viewer("20001", result_queues)

    while True:
        continue


if __name__ == "__main__":
    run_web_demo()
