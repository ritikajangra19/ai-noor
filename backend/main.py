import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription
)

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

@app.post("/offer")
async def offer(data: dict):

    print("=" * 50)
    print(data)
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

