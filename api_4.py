import os
import cv2
import uuid
import base64
import asyncio
import logging
import threading

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ==================================================
# IMPORT YOUR MUSETALK MODULE
# ==================================================

from scripts.realtime_inference import (
    Avatar,
    STREAM_QUEUE,
    initialize_models,
    create_avatar
)

# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("NOOR")

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# GLOBALS
# ==================================================

clients = set()

AVATAR = None

# ==================================================
# WEBSOCKET MANAGER
# ==================================================

async def broadcast_frame(frame):

    if len(clients) == 0:
        return

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if not success:
        return

    jpg_bytes = encoded.tobytes()

    payload = base64.b64encode(
        jpg_bytes
    ).decode()

    disconnected = []

    for ws in clients:

        try:

            await ws.send_text(payload)

        except:

            disconnected.append(ws)

    for ws in disconnected:

        clients.remove(ws)

# ==================================================
# STREAM LOOP
# ==================================================

async def stream_loop():

    logger.info(
        "[STREAM] loop started"
    )

    while True:

        try:

            frame = STREAM_QUEUE.get(
                timeout=1
            )

            logger.info(
                "[FRAME] sending frame"
            )

            await broadcast_frame(
                frame
            )

        except:

            await asyncio.sleep(
                0.01
            )

# ==================================================
# STARTUP
# ==================================================

@app.on_event("startup")
async def startup():

    logger.info(
        "[STARTUP] backend started"
    )

    asyncio.create_task(
        stream_loop()
    )

# ==================================================
# WEBSOCKET
# ==================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    clients.add(websocket)

    logger.info(
        f"[WS] client connected "
        f"total={len(clients)}"
    )

    try:

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        clients.remove(websocket)

        logger.info(
            f"[WS] client disconnected "
            f"total={len(clients)}"
        )

# ==================================================
# MUSETALK WORKER
# ==================================================

def run_musetalk_job(
    audio_path
):

    try:

        logger.info(
            "[JOB] started"
        )

        logger.info(
            f"[AUDIO] {audio_path}"
        )

        AVATAR.inference(
            audio_path=audio_path,
            out_vid_name=None,
            fps=25,
            skip_save_images=True
        )

        logger.info(
            "[JOB] completed"
        )

    except Exception as e:

        logger.exception(
            f"[JOB] failed {e}"
        )

# ==================================================
# UPLOAD AUDIO
# ==================================================

@app.post("/upload")

async def upload_audio(
    audio: UploadFile = File(...)
):

    job_id = str(
        uuid.uuid4()
    )[:8]

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    file_path = (
        f"uploads/{job_id}_"
        f"{audio.filename}"
    )

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(
            await audio.read()
        )

    logger.info(
        f"[UPLOAD] {file_path}"
    )

    thread = threading.Thread(
        target=run_musetalk_job,
        args=(file_path,),
        daemon=True
    )

    thread.start()

    return {
        "status": "started",
        "job_id": job_id
    }

# ==================================================
# HEALTH
# ==================================================

@app.get("/health")

def health():

    return {
        "status": "ok"
    }

@app.on_event("startup")
async def startup():

    initialize_models()

    global AVATAR

    AVATAR = create_avatar()

    print(
        "[STARTUP] Avatar ready"
    )