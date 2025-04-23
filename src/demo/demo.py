import argparse
import threading

from warboy import WARBOY_APP, get_demo_params_from_cfg, spawn_web_viewer


def run_warboy_app(warboy_app):
    t = threading.Thread(target=warboy_app)
    t.daemon = True
    t.start()
    return t.native_id


def run_web_demo(cfg_path="tests/warboy/cfg/demo_config/demo.yaml"):
    demo_params = get_demo_params_from_cfg(cfg_path)
    warboy_app = WARBOY_APP(demo_params, "web")
    run_warboy_app(warboy_app)
    result_queues = warboy_app.get_result_queues()
    spawn_web_viewer("20001", result_queues)

    while True:
        continue

def run_make_file(cfg_path="tests/warboy/cfg/demo_config/demo.yaml"):
    demo_params = get_demo_params_from_cfg(cfg_path)
    warboy_app = WARBOY_APP(demo_params, "file")
    
    warboy_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script")
    parser.add_argument("mode", choices=["web", "file"], help="Choose the mode to run")

    parser.add_argument(
        "--cfg-path",
        type=str,
        default="tests/warboy/cfg/demo_config/demo.yaml",
        help="Path to the configuration file",
    )

    args = parser.parse_args()

    if args.mode == "web":
        run_web_demo(args.cfg_path)
    elif args.mode == "file":
        run_make_file(args.cfg_path)
    else:
        print("Invalid mode. Choose 'web' or 'file'.")
        exit(1)
