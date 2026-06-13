import os
import cv2
import time
import uuid
import queue
import asyncio
import threading
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# GLOBAL STATE
# =====================================================

FRAME_QUEUE = queue.Queue(maxsize=500)

CURRENT_JOB = None

# =====================================================
# LOGGER
# =====================================================

def log(message):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )

# =====================================================
# MUSE WORKER
# =====================================================

class MuseTalkWorker:

    def __init__(self):
        log("Loading MuseTalk...")

        # TODO
        # load models here

        log("MuseTalk loaded successfully")

    def process_audio(self, audio_path):

        log(f"Audio received: {audio_path}")

        FRAME_QUEUE.queue.clear()

        # ==================================================
        # REPLACE THIS WITH REAL MUSETALK
        # ==================================================

        for i in range(150):

            frame = np.zeros(
                (720, 1280, 3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                f"Noor Avatar Frame {i}",
                (150, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255,255,255),
                4
            )

            cv2.putText(
                frame,
                f"Audio Processing...",
                (150,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            FRAME_QUEUE.put(frame)

            log(
                f"Generated frame {i}"
            )

            time.sleep(0.04)

        log("MuseTalk job completed")

worker = MuseTalkWorker()

# =====================================================
# AUDIO UPLOAD
# =====================================================

@app.post("/upload-audio")
async def upload_audio(
    audio: UploadFile = File(...)
):

    global CURRENT_JOB

    job_id = str(uuid.uuid4())[:8]

    os.makedirs("uploads", exist_ok=True)

    path = f"uploads/{job_id}_{audio.filename}"

    with open(path, "wb") as f:
        f.write(await audio.read())

    log(f"[{job_id}] Audio uploaded")

    CURRENT_JOB = job_id

    threading.Thread(
        target=worker.process_audio,
        args=(path,),
        daemon=True
    ).start()

    return {
        "job_id": job_id
    }

# =====================================================
# STREAM FRAMES
# =====================================================

def frame_generator():

    log("Frontend connected to stream")

    while True:

        try:

            frame = FRAME_QUEUE.get(timeout=5)

            _, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

        except Exception:

            continue

@app.get("/stream")
async def stream():

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
def health():

    return JSONResponse(
        {
            "status": "running"
        }
    )