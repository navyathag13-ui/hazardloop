"""OpenCV 5 vision stage: detects heads and checks for hard-hat PPE compliance.

Uses OpenCV 5's DNN-based FaceDetectorYN (from the official OpenCV Zoo) to
locate heads, then a color-heuristic check on the region above each head for
hard-hat colors. OpenCV 5 dropped the legacy HOGDescriptor/CascadeClassifier
people/face detectors from the Python bindings in favor of DNN detectors like
this one.
"""
from pathlib import Path

import cv2
import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet.onnx"

# Bright hard-hat colors in HSV: yellow, orange, white, red
_HAT_COLOR_RANGES = [
    ((20, 100, 100), (35, 255, 255)),   # yellow
    ((5, 100, 100), (19, 255, 255)),    # orange
    ((0, 0, 200), (180, 30, 255)),      # white
    ((0, 100, 100), (4, 255, 255)),     # red (low hue)
    ((170, 100, 100), (180, 255, 255)),  # red (high hue wrap)
]

_detector = cv2.FaceDetectorYN_create(
    str(_MODEL_PATH), "", (320, 320), score_threshold=0.6
)


def detect_heads(frame: np.ndarray):
    """Returns list of (x, y, w, h) head bounding boxes."""
    h, w = frame.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(frame)
    if faces is None:
        return []
    return [tuple(int(v) for v in face[:4]) for face in faces]


def has_hard_hat(frame: np.ndarray, head_box) -> bool:
    """Checks the region above a detected head for hard-hat colors."""
    x, y, w, h = head_box
    frame_h, frame_w = frame.shape[:2]

    # Hat sits above and slightly overlapping the top of the head box.
    hat_y0 = max(0, y - int(h * 0.6))
    hat_y1 = min(frame_h, y + int(h * 0.3))
    hat_x0 = max(0, x - int(w * 0.15))
    hat_x1 = min(frame_w, x + w + int(w * 0.15))

    hat_roi = frame[hat_y0:hat_y1, hat_x0:hat_x1]
    if hat_roi.size == 0:
        return False

    hsv = cv2.cvtColor(hat_roi, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _HAT_COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)

    coverage = cv2.countNonZero(mask_total) / mask_total.size
    return coverage > 0.15  # >15% of the hat region matches a hard-hat color


def run_inspection(frame: np.ndarray):
    """Runs the full OpenCV 5 vision stage and returns structured per-person results."""
    heads = detect_heads(frame)
    results = []
    for box in heads:
        compliant = has_hard_hat(frame, box)
        results.append({
            "bbox": box,
            "ppe_compliant": compliant,
        })
    return results
