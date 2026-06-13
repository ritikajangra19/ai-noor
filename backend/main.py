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
            is_elevenlabs = message.get("elevenlabs", False)
            print(f"[WS] Decoded message - User text: '{user_text}', ElevenLabs: {is_elevenlabs}")
            
            if is_elevenlabs:
                audio_path = "data/audio/11lab-audio-noor.mp3"
                response_text = "Playing pre-saved ElevenLabs audio demo."
                print(f"[WS] ElevenLabs mode selected. Using pre-saved audio: {audio_path}")
                if not os.path.exists(audio_path):
                    # Copy fallback from assets if not exists
                    assets_audio = "assets/11lab-audio-noor.mp3"
                    print(f"[WS] Pre-saved audio not found. Copying from {assets_audio}...")
                    if os.path.exists(assets_audio):
                        os.makedirs("data/audio", exist_ok=True)
                        import shutil
                        shutil.copy(assets_audio, audio_path)
            else:
                response_text = user_text
                audio_filename = f"tts_{uuid.uuid4().hex}.mp3"
                audio_path = os.path.join(UPLOAD_DIR, audio_filename)
                
                tts_text = response_text.replace("species", "spee-sheez")
                print(f"[WS] Edge-TTS mode selected. Generating audio for: '{tts_text}' -> file: {audio_path}")
                communicate = edge_tts.Communicate(tts_text, "en-US-JennyNeural")
                await communicate.save(audio_path)
                print(f"[WS] Edge-TTS audio generated successfully.")
            
            # Extract Whisper features
            print(f"[WS] Extracting Whisper audio features from {audio_path}...")
            audio_processor = models["audio_processor"]
            device = models["device"]
            weight_dtype = models["weight_dtype"]
            whisper = models["whisper"]
            pe = models["pe"]
            unet = models["unet"]
            vae = models["vae"]
            timesteps = models["timesteps"]
            
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=25,
                audio_padding_length_left=2,
                audio_padding_length_right=2
            )
            
            video_num = len(whisper_chunks)
            print(f"[WS] Whisper feature extraction complete. Total chunks/frames: {video_num}")
            
            # Load audio bytes to send to frontend
            print(f"[WS] Loading audio bytes to base64...")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                
            # Send initial metadata
            print(f"[WS] Sending start signal payload to client...")
            await websocket.send_text(json.dumps({
                "type": "start",
                "text": response_text,
                "audio": f"data:audio/mp3;base64,{audio_base64}",
                "total_frames": video_num
            }))
            
            batch_size = 8
            print(f"[WS] Initializing datagen batch generator with batch_size={batch_size}...")
            gen = datagen(
                whisper_chunks,
                models["input_latent_list_cycle"],
                batch_size
            )
            
            frame_idx = 0
            coord_list_cycle = models["coord_list_cycle"]
            frame_list_cycle = models["frame_list_cycle"]
            mask_list_cycle = models["mask_list_cycle"]
            mask_coords_list_cycle = models["mask_coords_list_cycle"]
            
            print(f"[WS] Starting MuseTalk model inference & frames streaming loop...")
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon = vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    bbox = coord_list_cycle[frame_idx % (len(coord_list_cycle))]
                    ori_frame = copy.deepcopy(frame_list_cycle[frame_idx % (len(frame_list_cycle))])
                    x1, y1, x2, y2 = bbox
                    
                    try:
                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception as e:
                        print(f"[WS] Warning: Failed to resize frame {frame_idx}: {e}")
                        frame_idx += 1
                        continue
                        
                    mask = mask_list_cycle[frame_idx % (len(mask_list_cycle))]
                    mask_crop_box = mask_coords_list_cycle[frame_idx % (len(mask_coords_list_cycle))]
                    
                    combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                    
                    _, buffer = cv2.imencode('.jpg', combine_frame)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")
                    
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "index": frame_idx,
                        "image": f"data:image/jpeg;base64,{frame_base64}"
                    }))
                    
                    if frame_idx % 25 == 0 or frame_idx == video_num - 1:
                        print(f"[WS] Sent frame {frame_idx + 1}/{video_num} to client.")
                        
                    await asyncio.sleep(0.002)
                    frame_idx += 1
                    
            await websocket.send_text(json.dumps({"type": "end"}))
            print("[WS] Sent end signal to client. Successfully streamed all video frames.")
            
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
