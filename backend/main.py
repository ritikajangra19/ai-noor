import sys
import os
import asyncio
import pickle
import glob
import uuid
import base64
import json
import torch
import cv2
import numpy as np
import edge_tts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.chdir(PARENT_DIR)
print(f"[Backend] Working directory: {os.getcwd()}")

from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image_blending
from transformers import WhisperModel

app = FastAPI(title="Noor AI WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory="data/audio"), name="audio")
app.mount("/full_imgs", StaticFiles(directory="results/v15/avatars/avator_1/full_imgs"), name="full_imgs")

models = {}

# Quality 82 balances visual fidelity vs bandwidth for streaming
_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 82]
# 16 gives better GPU utilization without OOM risk on most cards
_BATCH_SIZE = 16


@app.on_event("startup")
async def startup_event():
    print("Loading MuseTalk models and pre-caching avatar...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    use_float16 = torch.cuda.is_available()
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    whisper = WhisperModel.from_pretrained("models/whisper")
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

    with open(os.path.join(avatar_dir, "coords.pkl"), "rb") as f:
        coord_list_cycle = pickle.load(f)
    with open(os.path.join(avatar_dir, "mask_coords.pkl"), "rb") as f:
        mask_coords_list_cycle = pickle.load(f)

    input_img_list = sorted(glob.glob(os.path.join(avatar_dir, "full_imgs", "*.png")))
    frame_list_cycle = [cv2.imread(p) for p in input_img_list]
    print(f"Loaded {len(frame_list_cycle)} background frames into RAM.")

    input_mask_list = sorted(glob.glob(os.path.join(avatar_dir, "mask", "*.png")))
    mask_list_cycle = [cv2.imread(p) for p in input_mask_list]
    print(f"Loaded {len(mask_list_cycle)} mask images into RAM.")

    models.update({
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "timesteps": timesteps,
        "weight_dtype": weight_dtype,
        "audio_processor": audio_processor,
        "whisper": whisper,
        "fp": fp,
        "device": device,
        "input_latent_list_cycle": input_latent_list_cycle,
        "coord_list_cycle": coord_list_cycle,
        "mask_coords_list_cycle": mask_coords_list_cycle,
        "frame_list_cycle": frame_list_cycle,
        "mask_list_cycle": mask_list_cycle,
    })
    print("All models and avatar cache loaded successfully.")


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


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")

    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            user_text = message.get("text", "")
            is_elevenlabs = message.get("elevenlabs", False)

            # ── Step 1: TTS ──────────────────────────────────────────────────
            if is_elevenlabs:
                audio_path = "data/audio/11lab-audio-noor.mp3"
                response_text = "Playing pre-saved ElevenLabs audio demo."
                if not os.path.exists(audio_path):
                    assets_audio = "assets/11lab-audio-noor.mp3"
                    if os.path.exists(assets_audio):
                        import shutil
                        os.makedirs("data/audio", exist_ok=True)
                        shutil.copy(assets_audio, audio_path)
            else:
                response_text = user_text
                audio_filename = f"tts_{uuid.uuid4().hex}.mp3"
                audio_path = os.path.join(UPLOAD_DIR, audio_filename)
                tts_text = response_text.replace("species", "spee-sheez")
                communicate = edge_tts.Communicate(tts_text, "en-US-JennyNeural")
                await communicate.save(audio_path)
                print(f"[WS] TTS saved: {audio_path}")

            # ── Step 2: Whisper feature extraction in thread pool ────────────
            # run_in_executor prevents GPU/CPU work from blocking the event
            # loop so WebSocket keep-alives and disconnect events still fire.
            audio_processor = models["audio_processor"]
            device = models["device"]
            weight_dtype = models["weight_dtype"]
            whisper_model = models["whisper"]

            whisper_input_features, librosa_length = await loop.run_in_executor(
                None,
                lambda: audio_processor.get_audio_feature(
                    audio_path, weight_dtype=weight_dtype
                ),
            )

            def _extract_chunks():
                with torch.no_grad():
                    return audio_processor.get_whisper_chunk(
                        whisper_input_features,
                        device,
                        weight_dtype,
                        whisper_model,
                        librosa_length,
                        fps=25,
                        audio_padding_length_left=2,
                        audio_padding_length_right=2,
                    )

            whisper_chunks = await loop.run_in_executor(None, _extract_chunks)
            video_num = len(whisper_chunks)
            print(f"[WS] Whisper done — {video_num} frames to generate.")

            # ── Step 3: Send START signal with audio NOW (before UNet runs) ──
            # Client receives audio + frame count and can start buffering
            # immediately. Progressive frames arrive as UNet generates them.
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            await websocket.send_text(json.dumps({
                "type": "start",
                "text": response_text,
                "audio": f"data:audio/mp3;base64,{audio_b64}",
                "total_frames": video_num,
            }))

            # ── Step 4: UNet inference + progressive frame streaming ─────────
            pe = models["pe"]
            unet = models["unet"]
            vae = models["vae"]
            timesteps = models["timesteps"]
            coord_list_cycle = models["coord_list_cycle"]
            frame_list_cycle = models["frame_list_cycle"]
            mask_list_cycle = models["mask_list_cycle"]
            mask_coords_list_cycle = models["mask_coords_list_cycle"]

            gen = datagen(
                whisper_chunks,
                models["input_latent_list_cycle"],
                _BATCH_SIZE,
            )

            frame_idx = 0
            for whisper_batch, latent_batch in gen:
                # Capture loop variables for the closure
                wb, lb = whisper_batch, latent_batch

                def _infer(wb=wb, lb=lb):
                    with torch.no_grad():
                        af = pe(wb.to(device))
                        lb = lb.to(device=device, dtype=unet.model.dtype)
                        pred = unet.model(
                            lb, timesteps, encoder_hidden_states=af
                        ).sample
                        pred = pred.to(device=device, dtype=vae.vae.dtype)
                        return vae.decode_latents(pred)

                recon = await loop.run_in_executor(None, _infer)

                for res_frame in recon:
                    bbox = coord_list_cycle[frame_idx % len(coord_list_cycle)]
                    # .copy() avoids the overhead of copy.deepcopy for ndarray
                    ori_frame = frame_list_cycle[frame_idx % len(frame_list_cycle)].copy()
                    x1, y1, x2, y2 = bbox

                    try:
                        res_frame = cv2.resize(
                            res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                        )
                    except Exception:
                        frame_idx += 1
                        continue

                    mask = mask_list_cycle[frame_idx % len(mask_list_cycle)]
                    mask_crop_box = mask_coords_list_cycle[
                        frame_idx % len(mask_coords_list_cycle)
                    ]
                    combined = get_image_blending(
                        ori_frame, res_frame, bbox, mask, mask_crop_box
                    )

                    _, buf = cv2.imencode(".jpg", combined, _JPEG_PARAMS)
                    frame_b64 = base64.b64encode(buf).decode("utf-8")

                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "index": frame_idx,
                        "image": f"data:image/jpeg;base64,{frame_b64}",
                    }))

                    # asyncio.sleep(0) yields to the event loop with zero delay
                    # so disconnect/keepalive messages are processed between frames
                    await asyncio.sleep(0)
                    frame_idx += 1

            await websocket.send_text(json.dumps({"type": "end"}))
            print(f"[WS] Done — streamed {frame_idx} frames.")

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass
