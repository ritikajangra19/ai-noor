import sys
import os
import asyncio
import threading
import copy
import pickle
import glob
import uuid
import base64
import json
import torch
import cv2
import numpy as np
import edge_tts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path so we can import musetalk modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Force the working directory to be the parent directory (ai-noor root)
os.chdir(PARENT_DIR)
print(f"[Backend] Forced working directory to: {os.getcwd()}")

from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image_blending
from transformers import WhisperModel

import re

def split_into_sentences(text: str) -> list[str]:
    """Splits a long paragraph into individual sentences.
    If a sentence is extremely long (like a run-on text without punctuation),
    it splits it by clauses (commas) or by words to prevent blocking the GPU for too long.
    """
    # 1. Split by sentence end markers
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    sub_chunks = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
            
        # 2. If sentence is reasonably short, keep it intact
        if len(s) <= 150:
            sub_chunks.append(s)
            continue
            
        # 3. Try splitting by clause boundaries (comma, semicolon, colon, dash)
        clauses = re.split(r'(?<=[,;:—])\s+', s)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            if len(clause) <= 150:
                sub_chunks.append(clause)
                continue
                
            # 4. Hard-split by word boundaries to enforce max chunk size of ~120 chars
            # words = clause.split()
            # current_chunk = []
            # current_len = 0
            # for w in words:
            #     current_chunk.append(w)
            #     current_len += len(w) + 1
            #     if current_len >= 120:
            #         sub_chunks.append(" ".join(current_chunk))
            #         current_chunk = []
            #         current_len = 0
            # if current_chunk:
            #     sub_chunks.append(" ".join(current_chunk))

    # 5. Group very short adjacent chunks together for smooth natural speech flow
    combined = []
    current = ""
    for chunk in sub_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if current:
            # Only combine if the combined length is not too large
            if len(current) + len(chunk) + 1 <= 150:
                current = current + " " + chunk
            else:
                combined.append(current)
                current = chunk
        else:
            current = chunk
    if current:
        combined.append(current)
        
    return combined

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

# Global models dictionary
models = {}

@app.on_event("startup")
async def startup_event():
    print("Loading MuseTalk models and pre-caching avatar...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet_model_path = "models/musetalkV15/unet.pth"
    unet_config = "models/musetalkV15/musetalk.json"
    vae_type = "sd-vae"
    whisper_dir = "models/whisper"
    
    # Load core models
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path, 
        vae_type=vae_type,
        unet_config=unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    # Use float16 for high-speed inference on GPU
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
    
    # Load Whisper audio models
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Face parser
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    
    # Load avatar cache (avator_1 avatar)
    avatar_dir = "results/v15/avatars/avator_1"
    if not os.path.exists(avatar_dir):
        raise RuntimeError(f"Avatar cache at {avatar_dir} not found. Please run the material preparation script first!")
        
    latents_path = os.path.join(avatar_dir, "latents.pt")
    coords_path = os.path.join(avatar_dir, "coords.pkl")
    mask_coords_path = os.path.join(avatar_dir, "mask_coords.pkl")
    full_imgs_dir = os.path.join(avatar_dir, "full_imgs")
    mask_dir = os.path.join(avatar_dir, "mask")
    
    # Load pre-computed coordinates, latents and blend masks
    input_latent_list_cycle = torch.load(latents_path)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
        
    # Read background images into memory
    input_img_list = sorted(glob.glob(os.path.join(full_imgs_dir, '*.png')))
    frame_list_cycle = []
    print(f"Reading {len(input_img_list)} background images into memory...")
    for img_path in input_img_list:
        frame_list_cycle.append(cv2.imread(img_path))
        
    # Read mask images into memory
    input_mask_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    mask_list_cycle = []
    print(f"Reading {len(input_mask_list)} mask images into memory...")
    for mask_path in input_mask_list:
        mask_list_cycle.append(cv2.imread(mask_path))

    # Shift everything by 75 frames to start lipsync from frame 76
    SHIFT = 75
    if len(coord_list_cycle) > SHIFT:
        print(f"Shifting cached lists and latents by {SHIFT} frames to start lipsync from frame 76...")
        if isinstance(input_latent_list_cycle, torch.Tensor):
            input_latent_list_cycle = torch.cat([input_latent_list_cycle[SHIFT:], input_latent_list_cycle[:SHIFT]], dim=0)
        else:
            input_latent_list_cycle = input_latent_list_cycle[SHIFT:] + input_latent_list_cycle[:SHIFT]
            
        coord_list_cycle = coord_list_cycle[SHIFT:] + coord_list_cycle[:SHIFT]
        mask_coords_list_cycle = mask_coords_list_cycle[SHIFT:] + mask_coords_list_cycle[:SHIFT]
        frame_list_cycle = frame_list_cycle[SHIFT:] + frame_list_cycle[:SHIFT]
        mask_list_cycle = mask_list_cycle[SHIFT:] + mask_list_cycle[:SHIFT]
        
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
        "mask_list_cycle": mask_list_cycle
    })
    print("All MuseTalk models and avatar cached data loaded successfully!")


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("backend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h3>index.html not found: {str(e)}</h3>"


@app.get("/portal-resolver")
def portal_resolver():
    return {"status": "ok"}


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected.")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WS] Received message payload: {data}")
            message = json.loads(data)
            user_text = message.get("text", "")
            session_id = message.get("session_id", "")
            is_elevenlabs = message.get("elevenlabs", False)
            print(f"[WS] User text: '{user_text}', Session: '{session_id}', ElevenLabs: {is_elevenlabs}")

            sentences = [user_text] if is_elevenlabs else split_into_sentences(user_text)
            if not sentences:
                sentences = [user_text]
            print(f"[WS] Sentences ({len(sentences)}): {sentences}")

            # ── Step 1: Generate ALL TTS audio files concurrently ──────────────────
            # Edge-TTS is a pure network call with no GPU involvement, so all
            # sentences can be synthesized in parallel. This eliminates the serial
            # TTS wait that previously caused a silent gap between each chunk.
            async def gen_tts(sentence):
                if is_elevenlabs:
                    path = "data/audio/11lab-audio-noor.mp3"
                    if not os.path.exists(path):
                        src = "assets/11lab-audio-noor.mp3"
                        if os.path.exists(src):
                            import shutil
                            os.makedirs("data/audio", exist_ok=True)
                            shutil.copy(src, path)
                    return path, "Playing pre-saved ElevenLabs audio demo."
                fname = f"tts_{uuid.uuid4().hex}.mp3"
                path = os.path.join(UPLOAD_DIR, fname)
                tts_text = sentence.replace("species", "spee-sheez")
                communicate = edge_tts.Communicate(tts_text, "en-US-JennyNeural")
                await communicate.save(path)
                return path, sentence

            print(f"[WS] Generating TTS for {len(sentences)} sentence(s) concurrently...")
            tts_results = await asyncio.gather(*[gen_tts(s) for s in sentences])
            print("[WS] All TTS audio ready.")

            # ── Step 2: Pre-compute Whisper features for ALL chunks (GPU, sequential)
            # Done upfront so there is zero Whisper latency between chunks during
            # streaming — when chunk N finishes, chunk N+1 inference starts instantly.
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
                print(f"[WS] Whisper pre-compute for chunk {chunk_idx}...")
                wif, lib_len = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
                wchunks = audio_processor.get_whisper_chunk(
                    wif, device, weight_dtype, whisper, lib_len,
                    fps=25, audio_padding_length_left=2, audio_padding_length_right=2
                )
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                all_chunk_data.append({
                    "response_text": response_text,
                    "whisper_chunks": wchunks,
                    "audio_b64": audio_b64,
                    "video_num": len(wchunks),
                })
                await asyncio.sleep(0)
            print(f"[WS] Whisper pre-compute done for all {len(all_chunk_data)} chunk(s).")

            # ── Step 3: Stream inference frames for all chunks back-to-back ─────────
            # Because TTS and Whisper are already done, the backend transitions from
            # chunk N to chunk N+1 with zero additional blocking — no inter-chunk gap.
            for chunk_idx, cd in enumerate(all_chunk_data):
                video_num = cd["video_num"]
                print(f"[WS] Streaming chunk {chunk_idx}: {video_num} frames | '{cd['response_text'][:50]}'")

                await websocket.send_text(json.dumps({
                    "type": "chunk_start",
                    "session_id": session_id,
                    "chunk_index": chunk_idx,
                    "text": cd["response_text"],
                    "audio": f"data:audio/mp3;base64,{cd['audio_b64']}",
                    "total_frames": video_num
                }))

                gen = datagen(cd["whisper_chunks"], models["input_latent_list_cycle"], 16)
                frame_idx = 0

                for whisper_batch, latent_batch in gen:
                    audio_feature_batch = pe(whisper_batch.to(device))
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                    recon = vae.decode_latents(pred_latents)

                    for res_frame in recon:
                        bbox      = coord_list_cycle[frame_idx % len(coord_list_cycle)]
                        ori_frame = copy.deepcopy(frame_list_cycle[frame_idx % len(frame_list_cycle)])
                        x1, y1, x2, y2 = bbox
                        try:
                            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        except Exception as e:
                            print(f"[WS] Warning: Failed to resize frame {frame_idx}: {e}")
                            frame_idx += 1
                            continue

                        mask          = mask_list_cycle[frame_idx % len(mask_list_cycle)]
                        mask_crop_box = mask_coords_list_cycle[frame_idx % len(mask_coords_list_cycle)]
                        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

                        _, buffer = cv2.imencode('.jpg', combine_frame)
                        frame_base64 = base64.b64encode(buffer).decode("utf-8")

                        await websocket.send_text(json.dumps({
                            "type": "frame",
                            "session_id": session_id,
                            "chunk_index": chunk_idx,
                            "index": frame_idx,
                            "image": f"data:image/jpeg;base64,{frame_base64}"
                        }))

                        if frame_idx % 25 == 0 or frame_idx == video_num - 1:
                            print(f"[WS] Chunk {chunk_idx}: sent frame {frame_idx + 1}/{video_num}")

                        await asyncio.sleep(0)
                        frame_idx += 1

                await websocket.send_text(json.dumps({
                    "type": "chunk_end",
                    "session_id": session_id,
                    "chunk_index": chunk_idx
                }))
                print(f"[WS] Finished streaming chunk {chunk_idx}.")

            await websocket.send_text(json.dumps({"type": "end", "session_id": session_id}))
            print("[WS] Sent global end signal to client.")

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"[WS] Exception raised in WebSocket loop: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except Exception as send_err:
            print(f"Failed to send error details to client: {send_err}")
