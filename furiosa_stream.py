import multiprocessing as mp
import os
import sys

import fastapi
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import yaml

from video_utils.viewer import ResultViewer

app = FastAPI()


def get_params_from_cfg(cfg: str):
    num_channel = 0
    with open(cfg) as f:
        app_infos = yaml.load_all(f, Loader=yaml.FullLoader)
        output_paths = []
        for app_info in app_infos:
            model_config = open(app_info["model_config"])
            model_info = yaml.load(model_config, Loader=yaml.FullLoader)
            model_config.close()

            for video_path in app_info["video_path"]:
                video_name = (video_path.split('/')[-1]).split('.')[0]
                output_paths.append(os.path.join(app_info["output_path"], "output", video_name))

    return output_paths


VIEWER = ResultViewer(
    get_params_from_cfg(sys.argv[1]), full_grid_shape=(720, 1280), viewer=sys.argv[2]
)


@app.get("/")
def stream():
    return StreamingResponse(
        VIEWER.draw_img_to_grid_video(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    if sys.argv[2] == "fastAPI":
        uvicorn.run(
            app="furiosa_stream:app",
            host="0.0.0.0",
            port=20001,
            reload=False,
        )
    else:
        VIEWER.draw_img_to_grid_video()
