"""Sample each clip in data/videos/ at 1 frame per second into data/videos/frames/<id>_<second>.jpg.

Run:  python bench/sample_frames.py
"""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "data" / "videos"


def main():
    out = VIDEOS / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for clip in sorted(VIDEOS.glob("*.mp4")):
        cap = cv2.VideoCapture(str(clip))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t = 0
        while t * fps < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{clip.stem}_{t:02d}.jpg"), frame)
            t += 1
        print(clip.stem, t, "frames")


if __name__ == "__main__":
    main()
