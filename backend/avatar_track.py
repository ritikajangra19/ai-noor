from aiortc import VideoStreamTrack
from av import VideoFrame
import cv2
import os


class AvatarTrack(VideoStreamTrack):

    def __init__(self):
        super().__init__()

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

        ret, frame = self.cap.read()

        print("FRAME READ:", ret)

        if not ret:
            print("RESTART VIDEO")

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = self.cap.read()

        if frame is None:
            raise Exception(
                "Frame is None"
            )

        print(
            "FRAME SHAPE:",
            frame.shape
        )

        video_frame = VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame
