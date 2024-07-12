import threading

from warboy import WARBOY_APP, get_demo_params_from_cfg, spawn_web_viewer, MODEL_LIST


def run_warboy_app(warboy_app):
    t = threading.Thread(target=warboy_app)
    t.daemon = True
    t.start()
    return t.native_id


if __name__ == "__main__":
    print(MODEL_LIST)
    demo_params = get_demo_params_from_cfg("warboy/cfg/demo_config/demo.yaml")
    warboy_app = WARBOY_APP(demo_params)
    run_warboy_app(warboy_app)
    result_queues = warboy_app.get_result_queues()
    proc = spawn_web_viewer("20001", result_queues)

    while True:
        continue
