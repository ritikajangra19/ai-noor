import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# 1. Global Model Loader (Keep MuseTalk hot in GPU memory)
# -----------------------------------------------------------------
class MuseTalkPredictor:
    def __init__(self):
        # TODO: Initialize your MuseTalk UNet, Whisper/Audio models here ONCE.
        # e.g., self.unet = load_model("models/musetalkV15/unet.pth")
        print("MuseTalk Models loaded into GPU memory successfully.")

    def inference_chunk(self, audio_frame):
        """
        Receives an audio frame, runs MuseTalk inference for that slice,
        and returns a simulated or generated numpy image frame.
        """
        # TODO: Replace this placeholder with MuseTalk's internal frame generator logic
        # You will pass the audio features + your avatar reference image/features.
        
        # simulated_nd_array = self.model.predict(audio_frame)
        # return simulated_nd_array
        pass

# Initialize the predictor globally on startup
predictor = MuseTalkPredictor()
relay = MediaRelay()

# -----------------------------------------------------------------
# 2. Custom WebRTC Track Transformer
# -----------------------------------------------------------------
class MuseTalkTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, audio_track):
        super().__init__()
        self.audio_track = audio_track

    async def recv(self):
        """
        This method is continuously called by aiortc to pull the next 
        video frame to send to the client.
        """
        try:
            # Get the next incoming real-time audio frame from the client
            audio_frame = await self.audio_track.recv()
            
            # 1. Run your MuseTalk model slice on this specific chunk of audio
            # NOTE: If your model takes longer than ~40ms, wrap this in asyncio.to_thread
            numpy_img = predictor.inference_chunk(audio_frame)
            
            # 2. Convert your model output (Numpy array) into a WebRTC compatible VideoFrame
            # (Assuming output is standard 25fps RGB format)
            new_frame = VideoFrame.from_ndarray(numpy_img, format="rgb24")
            
            # Maintain correct timing synchronization
            new_frame.pts = audio_frame.pts
            new_frame.time_base = audio_frame.time_base
            
            return new_frame

        except Exception as e:
            print(f"Streaming error or track ended: {e}")
            raise

# -----------------------------------------------------------------
# 3. WebRTC Offer Endpoint (Fixed for SDP Transceiver Sync)
# -----------------------------------------------------------------
@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()

    # CRITICAL FIX 1: Pre-create a video transceiver. 
    # This explicitly tells aiortc that the server intends to SEND video back, 
    # preventing the 'None is not in list' direction error during createAnswer.
    video_transceiver = pc.addTransceiver("video", direction="sendonly")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("Received client audio track, routing to MuseTalk...")
            
            # Create our video generator track fueled by the incoming audio track
            video_track = MuseTalkTransformTrack(relay.subscribe(track))
            
            # CRITICAL FIX 2: Instead of pc.addTrack(), replace the track 
            # inside our pre-allocated transceiver.
            video_transceiver.sender.replaceTrack(video_track)

    # Set remote description (Client's setup)
    await pc.setRemoteDescription(offer)
    
    # Generate answer description (Server's configuration)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    )