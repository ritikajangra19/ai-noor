import sys
import os
import asyncio
import threading
import queue
import copy
import pickle
import glob
import shutil
import yaml
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription
)

# Add parent directory to path so we can import musetalk and scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from transformers import WhisperModel
import scripts.realtime_inference as ri
from avatar_track import AvatarTrack

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()
UPLOAD_DIR = "data/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LAST_AUDIO_PATH = None
_streamer = None
streamer_lock = threading.Lock()


class MuseTalkStreamer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[MuseTalk] Initializing MuseTalkStreamer on device: {self.device}")
        
        self.unet_model_path = "models/musetalkV15/unet.pth"
        self.unet_config = "models/musetalkV15/musetalk.json"
        self.whisper_dir = "models/whisper"
        self.vae_type = "sd-vae"
        self.version = "v15"
        
        if not os.path.exists(self.unet_model_path):
            self.unet_model_path = "models/musetalk/pytorch_model.bin"
            self.unet_config = "models/musetalk/musetalk.json"
            self.version = "v1"
            
        print(f"[MuseTalk] Version selected: {self.version}")
        
        # Create mock args namespace for realtime_inference
        class MockArgs:
            version = "v15" if self.version == "v15" else "v1"
            extra_margin = 10
            parsing_mode = "jaw"
            audio_padding_length_left = 2
            audio_padding_length_right = 2
            skip_save_images = True
            ffmpeg_path = "ffmpeg"
            gpu_id = 0
            
        ri.args = MockArgs()
        ri.device = self.device
        
        # Load core weights
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.unet_model_path,
            vae_type=self.vae_type,
            unet_config=self.unet_config,
            device=self.device
        )
        self.timesteps = torch.tensor([0], device=self.device)

        if torch.cuda.is_available():
            self.pe = self.pe.half().to(self.device)
            self.vae.vae = self.vae.vae.half().to(self.device)
            self.unet.model = self.unet.model.half().to(self.device)
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        # Initialize audio and whisper
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Initialize Face Parsing
        if self.version == "v15":
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        else:
            self.fp = FaceParsing()
            
        # Bind to realtime_inference module namespace
        ri.vae = self.vae
        ri.unet = self.unet
        ri.pe = self.pe
        ri.timesteps = self.timesteps
        ri.whisper = self.whisper
        ri.audio_processor = self.audio_processor
        ri.fp = self.fp
        print("[MuseTalk] MuseTalkStreamer successfully loaded.")


def get_streamer():
    global _streamer
    with streamer_lock:
        if _streamer is None:
            _streamer = MuseTalkStreamer()
        return _streamer


class StreamingAvatar(ri.Avatar):
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, webrtc_queue: asyncio.Queue, loop):
        self.webrtc_queue = webrtc_queue
        self.loop = loop
        super().__init__(avatar_id, video_path, bbox_shift, batch_size, preparation)

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(f"[StreamingAvatar] Rendering {video_len} frames to WebRTC stream...")
        while True:
            if self.idx >= video_len:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print("[StreamingAvatar] Resize error:", e)
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # Safely push frame to asyncio queue in event loop thread
            self.loop.call_soon_threadsafe(self.webrtc_queue.put_nowait, combine_frame)
            self.idx = self.idx + 1


@app.get("/")
def health():
    return {"status": "ok", "loaded": _streamer is not None}


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    global LAST_AUDIO_PATH
    try:
        # Save uploaded audio
        audio_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Audio saved: {audio_path}")
        LAST_AUDIO_PATH = audio_path

        # Update realtime config yaml
        try:
            with open("configs/inference/realtime.yaml", "r") as f:
                config = yaml.safe_load(f)
            config["avator_1"]["audio_clips"] = {
                "audio_0": audio_path
            }
            config["avator_1"]["preparation"] = False
            with open("configs/inference/realtime.yaml", "w") as f:
                yaml.dump(config, f)
        except Exception as e:
            print("Config yaml update skipped/failed:", e)

        # Trigger in-process fast inference
        print("Starting in-process fast MuseTalk inference...")
        streamer = get_streamer() # ensure loaded
        
        avatar_id = "avator_1"
        base_path = f"./results/v15/avatars/{avatar_id}"
        preparation = not os.path.exists(base_path)
        
        # standard Avatar class to write files to disk
        avatar = ri.Avatar(
            avatar_id=avatar_id,
            video_path="data/video/yongen.mp4",
            bbox_shift=5,
            batch_size=8,
            preparation=preparation
        )
        
        avatar.inference(
            audio_path=audio_path,
            out_vid_name="output_video",
            fps=25,
            skip_save_images=False
        )

        output_vid = os.path.join(avatar.video_out_path, "output_video.mp4")
        if not os.path.exists(output_vid):
            raise Exception("No video generated by MuseTalk")

        print(f"In-process MuseTalk completed. Returning: {output_vid}")
        return FileResponse(
            path=output_vid,
            media_type="video/mp4",
            filename=os.path.basename(output_vid)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/offer")
async def offer(data: dict):
    print("=" * 50)
    print("SDP OFFER RECEIVED")
    print("=" * 50)
    
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=data["sdp"],
            type=data["type"]
        )
    )

    audio_path = data.get("audio_path")
    if not audio_path:
        audio_path = LAST_AUDIO_PATH

    # If audio is available, spin up the real-time MuseTalk streaming track
    if audio_path and os.path.exists(audio_path):
        print(f"Initiating realtime MuseTalk avatar stream for audio: {audio_path}")
        webrtc_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        # Load/get models
        streamer = get_streamer()
        
        avatar_id = "avator_1"
        base_path = f"./results/v15/avatars/{avatar_id}"
        preparation = not os.path.exists(base_path)
        
        # Initialize StreamingAvatar subclass
        avatar = StreamingAvatar(
            avatar_id=avatar_id,
            video_path="data/video/yongen.mp4",
            bbox_shift=5,
            batch_size=8,
            preparation=preparation,
            webrtc_queue=webrtc_queue,
            loop=loop
        )

        # Run MuseTalk loop inside background thread to keep FastAPI responsive
        def run_thread():
            try:
                avatar.inference(
                    audio_path=audio_path,
                    out_vid_name=None,
                    fps=25,
                    skip_save_images=True
                )
            except Exception as e:
                print("Error in background streaming thread:", e)
                
        t = threading.Thread(target=run_thread)
        t.start()
        
        # Add the real-time Queue-based track to peer connection
        pc.addTrack(AvatarTrack(frame_queue=webrtc_queue))
    else:
        # Fallback to default avatar looping (no audio)
        print("No audio input or path found. Streaming default avatar on loop.")
        pc.addTrack(AvatarTrack())

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
