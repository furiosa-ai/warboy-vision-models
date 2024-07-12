import asyncio
import uvicorn
import time
import cv2
import queue
import multiprocessing as mp

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from warboy.utils.handler import ImageHandler
from warboy.utils.monitor_npu import WARBOYDevice

app = FastAPI()
# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요한 도메인으로 변경 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static"))

app.state.result_queues = list()
app.state.warboy_device = None
app.state.fps = 0


'''
def getByteFrame():
    result_img_handler = ImageHandler()
    result_queus = app.state.result_queues
    for full_grid_img, fps in result_img_handler(result_queus):
        app.state.fps = fps
        _, full_grid_img = cv2.imencode(".jpg", full_grid_img)
        out_frame = full_grid_img.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(out_frame) + b"\r\n"
        )


@app.get("/video_feed")
async def video_stream():
    return StreamingResponse(
        getByteFrame(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
'''

@app.get("/channels")
def get_channels():
    return {"channels": [i for i in range(len(app.state.result_queues))] }

'''
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel):
    await websocket.accept()
    result_queue = app.state.result_queues[int(channel)]
    
    FPS = 0.0
    try:
        while True:
            while True:
                try:
                    out_frame, FPS = result_queue.get(False)
                    break
                except queue.Empty:
                    await asyncio.sleep(0)
                except QueueClosedError:
                    break
            #_, grid_img = cv2.imencode(".jpg", out_frame)
            #out_frame = grid_img.tobytes()
            await websocket.send_bytes(out_frame)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
'''

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    FPS = 0.0
    #result_queue = app.state.result_queues[int(channel)]
    try:
        while True:
            img_datas = {}
            for idx, result_queue in enumerate(app.state.result_queues):
                while True:
                    try:
                        out_frame, FPS = result_queue.get(False)
                        break
                    except queue.Empty:
                        await asyncio.sleep(0)
                    except QueueClosedError:
                        break
                img_datas[f"camera_{idx}"] = list(out_frame)
            await websocket.send_json(img_datas)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

async def get_device_info():
    if app.state.warboy_device is None:
        app.state.warboy_device = await WARBOYDevice.create()
    power, util, temp, se, devices = await app.state.warboy_device()
    return jsonable_encoder(
        {
            "power": power,
            "util": util,
            "temp": temp,
            "time": se,
            "devices": devices,
            "fps": f"{app.state.fps:.1f}",
        }
    )


@app.get("/chart_data")
async def get_warboy_status():
    start_time = time.time()
    warboy_status = await get_device_info()
    await asyncio.sleep(1 - (time.time() - start_time))
    return JSONResponse(content=warboy_status)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def run_viewer(*args, **kwargs):
    uvicorn.run(*args, **kwargs)


def spawn_web_viewer(port: str, result_queues):
    app.state.result_queues = result_queues
    viewer_proc = mp.Process(
        target=run_viewer,
        args=("warboy.viewer:app",),
        kwargs={"host": "0.0.0.0", "port": int(port), },
    )
    viewer_proc.start()
    return viewer_proc
