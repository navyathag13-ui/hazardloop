"""HazardLoop API: see it, decide, act.

POST /inspect with an image runs the OpenCV 5 stage and, when a vision model
is configured, the vision LLM stage, then the agent decision that cross-checks
the two. The result can be logged to AWS S3.

  /inspect?mode=opencv   OpenCV stage and the rule-based decision only
  /inspect               vision LLM + OpenCV cross-check (falls back to
                         OpenCV with human review when no model is set up)
"""
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app import vision_llm  # noqa: E402
from app.agent import cross_check, decide  # noqa: E402
from app.aws import log_inspection  # noqa: E402
from app.detector import run_inspection  # noqa: E402

app = FastAPI(title="HazardLoop", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok", "vision_model": bool(vision_llm.get_client()[0])}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), mode: str = Query("cross_check", pattern="^(cross_check|opencv)$")):
    image_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image."})

    people = run_inspection(frame)
    result = {"people": people}
    if mode == "opencv":
        result["decision"] = {**decide(people), "source": "opencv"}
    else:
        counts, meta = vision_llm.count_ppe(image_bytes)
        result["vision_llm"] = {"counts": counts.model_dump() if counts else None, "meta": meta.model_dump()}
        result["decision"] = cross_check(people, counts, meta.validated)

    result["s3_key"] = log_inspection(image_bytes, result)
    return result
