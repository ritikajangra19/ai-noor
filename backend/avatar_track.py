import asyncio
from aiortc import VideoStreamTrack
from av import VideoFrame
import cv2
import os
import numpy as np


class AvatarTrack(VideoStreamTrack):

    def __init__(self, frame_queue: asyncio.Queue = None):
        super().__init__()
        self.frame_queue = frame_queue
        self.last_frame = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(current_dir, "avatar.mp4")

        print("VIDEO PATH:", video_path)

        self.cap = cv2.VideoCapture(video_path)

        print(
            "VIDEO OPENED:",
            self.cap.isOpened()
        )

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # If a frame queue is provided, try to pull frames from it in real-time
        if self.frame_queue is not None:
            try:
                # 40ms timeout for 25 FPS stream consistency
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.04)
                self.last_frame = frame
            except asyncio.TimeoutError:
                # Fallback to the last successfully read frame if queue is temporarily starved
                frame = self.last_frame
        else:
            # Standard looping behavior from file
            ret, frame = self.cap.read()
            print("FRAME READ:", ret)
            if not ret:
                print("RESTART VIDEO")
                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )
                ret, frame = self.cap.read()
            self.last_frame = frame

        if frame is None:
            # Hard fallback: black frame
            frame = np.zeros((256, 256, 3), dtype=np.uint8)

        # Print debug shapes
        # print("FRAME SHAPE:", frame.shape)

        video_frame = VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame
