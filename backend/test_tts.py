import edge_tts
import asyncio
import os

async def main():
    os.makedirs("data/audio", exist_ok=True)
    communicate = edge_tts.Communicate("Hello from zoo!", "en-US-JennyNeural")
    await communicate.save("data/audio/test_edge.mp3")
    print("TTS SUCCESS!")

if __name__ == "__main__":
    asyncio.run(main())
