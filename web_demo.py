import threading

from warboy.cfg import get_demo_params_from_cfg
from warboy.warboy_api import WARBOY_APP
from warboy.viewer import spawn_web_viewer


def run_warboy_app(warboy_app):
    t = threading.Thread(target=warboy_app.run_application)
    t.daemon = True
    t.start()
    return t.native_id


if __name__ == "__main__":
    demo_params = get_demo_params_from_cfg("warboy/cfg/demo_config/demo.yaml")
    warboy_app = WARBOY_APP(demo_params)
    run_warboy_app(warboy_app)
    proc = spawn_web_viewer("20001", warboy_app.get_result_queues())
    while True:
        continue
