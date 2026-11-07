import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.vision_llm import PPECounts, VisionMeta

client = TestClient(main.app)


def jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    return buf.tobytes()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_bad_image_is_rejected():
    r = client.post("/inspect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_opencv_mode():
    r = client.post("/inspect?mode=opencv", files={"file": ("x.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["decision"]["action"] == "pass"


def test_cross_check_mode_uses_the_vision_model(monkeypatch):
    monkeypatch.setattr(main.vision_llm, "count_ppe", lambda b: (
        PPECounts(with_hardhat=1, without_hardhat=1, unclear_heads=0, needs_human_review=False),
        VisionMeta(validated=True, attempts=1)))
    body = client.post("/inspect", files={"file": ("x.jpg", jpeg(), "image/jpeg")}).json()
    assert body["decision"]["action"] == "escalate" and body["decision"]["source"] == "vision_llm"


def test_no_vision_model_falls_back_to_review(monkeypatch):
    monkeypatch.setattr(main.vision_llm, "count_ppe", lambda b: (None, VisionMeta(validated=False, attempts=0)))
    body = client.post("/inspect", files={"file": ("x.jpg", jpeg(), "image/jpeg")}).json()
    assert body["decision"]["needs_human_review"] and body["decision"]["source"] == "opencv"


def test_unknown_mode_is_rejected():
    r = client.post("/inspect?mode=magic", files={"file": ("x.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422
