import os
import subprocess
import sys

import cv2
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

HOME_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(HOME_DIR)
from utils.parse_params import get_output_paths
from utils.result_img_process import ImageMerger

app = FastAPI()
CFG = sys.argv[1]
VIEWER = sys.argv[2]
MERGER = ImageMerger()


@app.get("/")
async def stream():
    def getByteFrame():
        output_paths = get_output_paths(CFG)
        for frame in MERGER(output_paths, full_grid_shape=(720, 1280)):
            ret, out_img = cv2.imencode(".jpg", frame)
            out_frame = out_img.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n"
            )

    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    if VIEWER == "fastAPI":
        uvicorn.run(app="stream:app", host="0.0.0.0", port=20001, reload=False)
    else:
        result_path = "output_"
        if VIEWER == "file":
            if os.path.exists(result_path):
                subprocess.run(["rm", "-rf", result_path])
            os.makedirs(result_path)
        output_paths = get_output_paths(CFG)
        for i, frame in enumerate(MERGER(output_paths, full_grid_shape=(720, 1280))):
            if VIEWER == "file":
                result_img_path = os.path.join(result_path, "%010d.bmp" % i)
                cv2.imwrite(result_img_path, frame)
            elif VIEWER == "open-cv":
                cv2.imshow("demo", frame)
                if cv2.waitKey(33) & 0xFF == ord("q"):
                    break
