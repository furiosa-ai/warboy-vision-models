import asyncio
import uvicorn
import time
import cv2
import multiprocessing as mp

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from warboy.utils.handler import ImageHandler
from warboy.utils.monitor_npu import WARBOYDevice

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"))
app.state.result_queues = list()
app.state.warboy_device = None


def getByteFrame():
    result_img_handler = ImageHandler()
    result_queus = app.state.result_queues
    for full_grid_img in result_img_handler(result_queus):
        _, full_grid_img = cv2.imencode(".jpg", full_grid_img)
        out_frame = full_grid_img.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n")

@app.get("/video_feed")
async def video_stream():
    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def get_device_info():
    if app.state.warboy_device is None:
        app.state.warboy_device = await WARBOYDevice.create()
    power, util, temp, se, devices = await app.state.warboy_device()
    return jsonable_encoder(
        {"power": power, "util": util, "temp": temp, "time": se, "devices": devices}
    )

@app.get("/chart_data")
async def get_warboy_status():
    start_time = time.time()
    warboy_status = await get_device_info()
    await asyncio.sleep(1 - (time.time()-start_time))
    return JSONResponse(content=warboy_status)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def run_viewer(*args, **kwargs):
    uvicorn.run(*args, **kwargs)

def spawn_web_viewer(port: str, result_queues):
    app.state.result_queues = result_queues
    viewer_proc = mp.Process(
        target = run_viewer,
        args=("warboy.viewer:app",),
        kwargs={
            "host": "0.0.0.0",
            "port": int(port),
        },
    )
    viewer_proc.start()
    return viewer_proc

