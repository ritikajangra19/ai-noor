import sys
import os
import asyncio
import copy
import pickle
import glob
import uuid
import base64
import json
import fractions
import torch
import cv2
import numpy as np
import edge_tts

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from av import VideoFrame

# ---------------------------------------------------------------------------
# Path bootstrap — keep working dir at the ai-noor root so all relative
# paths in musetalk (models/, results/, …) resolve correctly.
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
os.chdir(PARENT_DIR)
print(f"[Backend] Working directory: {os.getcwd()}")

from musetalk.utils.utils        import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending     import get_image_blending
from transformers                import WhisperModel

import re


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list:
    """Split a paragraph into sentence-sized chunks (≤ 150 chars)."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    sub_chunks = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(s) <= 150:
            sub_chunks.append(s)
            continue
        for clause in re.split(r'(?<=[,;:—])\s+', s):
            clause = clause.strip()
            if clause:
                sub_chunks.append(clause)

    combined, current = [], ""
    for chunk in sub_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if current and len(current) + len(chunk) + 1 <= 150:
            current = current + " " + chunk
        else:
            if current:
                combined.append(current)
            current = chunk
    if current:
        combined.append(current)
    return combined


# ---------------------------------------------------------------------------
# WebRTC video track
# ---------------------------------------------------------------------------

class FrameVideoTrack(VideoStreamTrack):
    """
    Pulls raw BGR numpy frames from an asyncio.Queue and emits them at
    exactly FPS via WebRTC.  Uses get_nowait() so the inference loop is
    never blocked — if the queue is empty the last good frame is frozen.
    """
    kind       = "video"
    FPS        = 25
    CLOCK_RATE = 90000          # standard RTP clock for video

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self._queue:      asyncio.Queue       = queue
        self._pts:        int                 = 0
        self._last_frame: Optional[np.ndarray] = None
        self._start:      Optional[float]     = None

    async def recv(self) -> VideoFrame:
        loop = asyncio.get_running_loop()
        now  = loop.time()

        if self._start is None:
            self._start = now

        # Pace output at exactly FPS — sleep only if we're ahead of schedule
        target = self._start + self._pts / self.FPS
        wait   = target - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)

        # Non-blocking dequeue: freeze on last frame rather than stalling
        try:
            frame_bgr        = self._queue.get_nowait()
            self._last_frame = frame_bgr
        except asyncio.QueueEmpty:
            frame_bgr = (
                self._last_frame
                if self._last_frame is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )

        av_frame            = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        av_frame.pts        = self._pts * (self.CLOCK_RATE // self.FPS)
        av_frame.time_base  = fractions.Fraction(1, self.CLOCK_RATE)
        self._pts          += 1
        return av_frame


# ---------------------------------------------------------------------------
# Per-connection session state
# ---------------------------------------------------------------------------

class SessionState:
    def __init__(self, session_id: str):
        self.session_id:  str                          = session_id
        self.pc:          Optional[RTCPeerConnection]  = None
        self.frame_queue: asyncio.Queue                = asyncio.Queue()
        self.data_channel                              = None   # aiortc DataChannel

    def send_control(self, message: dict) -> None:
        """Send a JSON control message to the client via the WebRTC data channel."""
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(json.dumps(message))


# session_id -> SessionState
sessions: dict = {}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Noor AI WebRTC Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/audio",     StaticFiles(directory="data/audio"),                              name="audio")
app.mount("/full_imgs", StaticFiles(directory="results/v15/avatars/avator_1/full_imgs"), name="full_imgs")

# Global model registry populated in startup
models: dict = {}


# ---------------------------------------------------------------------------
# Model loading (unchanged from original)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    print("Loading MuseTalk models and pre-caching avatar...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_model_path = "models/musetalkV15/unet.pth"
    unet_config     = "models/musetalkV15/musetalk.json"
    vae_type        = "sd-vae"
    whisper_dir     = "models/whisper"

    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type=vae_type,
        unet_config=unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    use_float16 = torch.cuda.is_available()
    if use_float16:
        pe.half(); vae.vae.half(); unet.model.half()
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    pe.to(device); vae.vae.to(device); unet.model.to(device)

    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    avatar_dir = "results/v15/avatars/avator_1"
    if not os.path.exists(avatar_dir):
        raise RuntimeError(
            f"Avatar cache at {avatar_dir} not found. "
            "Run the material preparation script first."
        )

    input_latent_list_cycle = torch.load(os.path.join(avatar_dir, "latents.pt"))
    with open(os.path.join(avatar_dir, "coords.pkl"),      "rb") as f:
        coord_list_cycle = pickle.load(f)
    with open(os.path.join(avatar_dir, "mask_coords.pkl"), "rb") as f:
        mask_coords_list_cycle = pickle.load(f)

    full_imgs_dir   = os.path.join(avatar_dir, "full_imgs")
    mask_dir        = os.path.join(avatar_dir, "mask")
    input_img_list  = sorted(glob.glob(os.path.join(full_imgs_dir, "*.png")))
    input_mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    print(f"Reading {len(input_img_list)} background images into memory...")
    frame_list_cycle = [cv2.imread(p) for p in input_img_list]
    print(f"Reading {len(input_mask_list)} mask images into memory...")
    mask_list_cycle  = [cv2.imread(p) for p in input_mask_list]

    # Shift by 75 frames to start lipsync from frame 76
    SHIFT = 75
    if len(coord_list_cycle) > SHIFT:
        print(f"Shifting cached lists by {SHIFT} frames...")
        if isinstance(input_latent_list_cycle, torch.Tensor):
            input_latent_list_cycle = torch.cat(
                [input_latent_list_cycle[SHIFT:], input_latent_list_cycle[:SHIFT]], dim=0
            )
        else:
            input_latent_list_cycle = input_latent_list_cycle[SHIFT:] + input_latent_list_cycle[:SHIFT]

        coord_list_cycle       = coord_list_cycle[SHIFT:]       + coord_list_cycle[:SHIFT]
        mask_coords_list_cycle = mask_coords_list_cycle[SHIFT:] + mask_coords_list_cycle[:SHIFT]
        frame_list_cycle       = frame_list_cycle[SHIFT:]       + frame_list_cycle[:SHIFT]
        mask_list_cycle        = mask_list_cycle[SHIFT:]        + mask_list_cycle[:SHIFT]

    models.update({
        "vae":                    vae,
        "unet":                   unet,
        "pe":                     pe,
        "timesteps":              timesteps,
        "weight_dtype":           weight_dtype,
        "audio_processor":        audio_processor,
        "whisper":                whisper,
        "fp":                     fp,
        "device":                 device,
        "input_latent_list_cycle": input_latent_list_cycle,
        "coord_list_cycle":        coord_list_cycle,
        "mask_coords_list_cycle":  mask_coords_list_cycle,
        "frame_list_cycle":        frame_list_cycle,
        "mask_list_cycle":         mask_list_cycle,
    })
    print("All models and avatar cache loaded.")


# ---------------------------------------------------------------------------
# Static endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("backend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h3>index.html not found: {e}</h3>"


@app.get("/portal-resolver")
def portal_resolver():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebRTC signaling
# ---------------------------------------------------------------------------

@app.post("/offer")
async def webrtc_offer(request: Request):
    """
    Client sends:  { sdp, type, session_id? }
    Server returns: { sdp, type, session_id }

    Client must:
      - addTransceiver('video', { direction: 'recvonly' })
      - createDataChannel('control')   ← receives control / audio JSON
      - pc.ontrack → attach stream to <video>
    """
    params     = await request.json()
    session_id = params.get("session_id") or str(uuid.uuid4())

    # Tear down any existing session for this id
    old = sessions.pop(session_id, None)
    if old and old.pc:
        await old.pc.close()

    state           = SessionState(session_id)
    sessions[session_id] = state

    pc = RTCPeerConnection(
        configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
        )
    )
    state.pc = pc

    # Attach the frame-streaming video track
    pc.addTrack(FrameVideoTrack(state.frame_queue))

    @pc.on("datachannel")
    def on_datachannel(channel):
        state.data_channel = channel
        print(f"[{session_id}] Data channel '{channel.label}' ready.")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        cstate = pc.connectionState
        print(f"[{session_id}] Connection → {cstate}")
        if cstate in ("failed", "closed"):
            sessions.pop(session_id, None)
            await pc.close()

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()

    # Wait for ICE gathering to finish so the answer SDP contains all candidates
    gather_done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_gather():
        if pc.iceGatheringState == "complete":
            gather_done.set()

    await pc.setLocalDescription(answer)

    if pc.iceGatheringState != "complete":
        try:
            await asyncio.wait_for(gather_done.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass   # use whatever candidates gathered so far

    print(f"[{session_id}] Offer answered. ICE state: {pc.iceGatheringState}")
    return JSONResponse({
        "sdp":        pc.localDescription.sdp,
        "type":       pc.localDescription.type,
        "session_id": session_id,
    })


# ---------------------------------------------------------------------------
# Generation trigger
# ---------------------------------------------------------------------------

@app.post("/generate")
async def webrtc_generate(request: Request):
    """
    Client sends: { session_id, text, elevenlabs? }

    Launches a background task that:
      1. Generates TTS (concurrent, I/O)
      2. Pre-computes Whisper features (GPU, sequential)
      3. Runs UNet inference and pushes raw BGR frames into the queue
         consumed by FrameVideoTrack — zero buffering, direct to WebRTC.
      Control/audio messages are sent via the WebRTC data channel.
    """
    params       = await request.json()
    session_id   = params.get("session_id", "")
    user_text    = params.get("text", "")
    is_elevenlabs = params.get("elevenlabs", False)

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Establish a WebRTC connection via POST /offer first.",
        )

    asyncio.create_task(
        generate_frames_task(session_id, user_text, is_elevenlabs)
    )
    return JSONResponse({"status": "generating"})


# ---------------------------------------------------------------------------
# Frame generation task (background)
# ---------------------------------------------------------------------------

async def generate_frames_task(
    session_id: str, user_text: str, is_elevenlabs: bool = False
) -> None:
    state = sessions.get(session_id)
    if not state:
        return

    print(f"[{session_id}] Generation started: '{user_text[:60]}'")

    try:
        # ── Step 1: TTS — all sentences concurrently (pure network I/O) ───────
        sentences = [user_text] if is_elevenlabs else split_into_sentences(user_text)
        if not sentences:
            sentences = [user_text]
        print(f"[{session_id}] {len(sentences)} sentence(s)")

        async def gen_tts(sentence: str):
            if is_elevenlabs:
                path = "data/audio/11lab-audio-noor.mp3"
                if not os.path.exists(path):
                    src = "assets/11lab-audio-noor.mp3"
                    if os.path.exists(src):
                        import shutil
                        os.makedirs("data/audio", exist_ok=True)
                        shutil.copy(src, path)
                return path, "Playing pre-saved ElevenLabs audio demo."
            fname      = f"tts_{uuid.uuid4().hex}.mp3"
            path       = os.path.join(UPLOAD_DIR, fname)
            tts_text   = sentence.replace("species", "spee-sheez")
            communicate = edge_tts.Communicate(tts_text, "en-US-JennyNeural")
            await communicate.save(path)
            return path, sentence

        print(f"[{session_id}] Generating TTS concurrently...")
        tts_results = await asyncio.gather(*[gen_tts(s) for s in sentences])
        print(f"[{session_id}] TTS done.")

        # ── Step 2: Whisper features — pre-compute all chunks upfront ─────────
        # Done before inference so there is zero inter-chunk Whisper latency.
        audio_processor        = models["audio_processor"]
        device                 = models["device"]
        weight_dtype           = models["weight_dtype"]
        whisper                = models["whisper"]
        pe                     = models["pe"]
        unet                   = models["unet"]
        vae                    = models["vae"]
        timesteps              = models["timesteps"]
        coord_list_cycle       = models["coord_list_cycle"]
        frame_list_cycle       = models["frame_list_cycle"]
        mask_list_cycle        = models["mask_list_cycle"]
        mask_coords_list_cycle = models["mask_coords_list_cycle"]

        all_chunk_data = []
        for chunk_idx, (audio_path, response_text) in enumerate(tts_results):
            print(f"[{session_id}] Whisper chunk {chunk_idx}...")
            wif, lib_len = audio_processor.get_audio_feature(
                audio_path, weight_dtype=weight_dtype
            )
            wchunks = audio_processor.get_whisper_chunk(
                wif, device, weight_dtype, whisper, lib_len,
                fps=25, audio_padding_length_left=2, audio_padding_length_right=2,
            )
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            all_chunk_data.append({
                "response_text":  response_text,
                "whisper_chunks": wchunks,
                "audio_b64":      audio_b64,
                "video_num":      len(wchunks),
            })
            await asyncio.sleep(0)   # yield between Whisper calls
        print(f"[{session_id}] Whisper pre-compute done ({len(all_chunk_data)} chunk(s)).")

        # ── Step 3: Inference — push BGR frames directly into the queue ───────
        # FrameVideoTrack.recv() drains this queue at 25 fps via WebRTC.
        # No JPEG encoding, no base64, no inter-frame sleep needed.
        for chunk_idx, cd in enumerate(all_chunk_data):
            video_num = cd["video_num"]
            print(
                f"[{session_id}] Chunk {chunk_idx}: "
                f"{video_num} frame(s) | '{cd['response_text'][:50]}'"
            )

            # Notify client: audio payload + metadata
            state.send_control({
                "type":         "chunk_start",
                "session_id":   session_id,
                "chunk_index":  chunk_idx,
                "text":         cd["response_text"],
                "audio":        f"data:audio/mp3;base64,{cd['audio_b64']}",
                "total_frames": video_num,
            })

            gen = datagen(
                cd["whisper_chunks"],
                models["input_latent_list_cycle"],
                batch_size=8,
            )

            frame_idx = 0
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch  = latent_batch.to(device=device, dtype=unet.model.dtype)
                pred_latents  = unet.model(
                    latent_batch, timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                pred_latents  = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon         = vae.decode_latents(pred_latents)

                for res_frame in recon:
                    bbox      = coord_list_cycle[frame_idx % len(coord_list_cycle)]
                    ori_frame = copy.deepcopy(frame_list_cycle[frame_idx % len(frame_list_cycle)])
                    x1, y1, x2, y2 = bbox

                    try:
                        res_frame = cv2.resize(
                            res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                        )
                    except Exception as resize_err:
                        print(f"[{session_id}] Resize error frame {frame_idx}: {resize_err}")
                        frame_idx += 1
                        continue

                    mask          = mask_list_cycle[frame_idx % len(mask_list_cycle)]
                    mask_crop_box = mask_coords_list_cycle[frame_idx % len(mask_coords_list_cycle)]
                    combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

                    # Push raw BGR frame — FrameVideoTrack encodes via H.264/VP8
                    state.frame_queue.put_nowait(combine_frame)

                    if frame_idx % 25 == 0 or frame_idx == video_num - 1:
                        print(
                            f"[{session_id}] Chunk {chunk_idx}: "
                            f"queued frame {frame_idx + 1}/{video_num}"
                        )
                    frame_idx += 1

                # Yield once per batch so the event loop can flush data-channel msgs
                await asyncio.sleep(0)

            state.send_control({
                "type":        "chunk_end",
                "session_id":  session_id,
                "chunk_index": chunk_idx,
            })
            print(f"[{session_id}] Chunk {chunk_idx} complete.")

        state.send_control({"type": "end", "session_id": session_id})
        print(f"[{session_id}] All chunks done.")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[{session_id}] Error: {exc}")
        state.send_control({"type": "error", "message": f"Server error: {exc}"})
