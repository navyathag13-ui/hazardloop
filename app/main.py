"""HazardLoop API: see it, decide, act.

POST /inspect with an image -> runs the OpenCV 5 vision stage, the agent
decision stage, optionally logs to AWS S3, and returns the full loop result.
"""
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.agent import decide
from app.aws import log_inspection
from app.detector import run_inspection

app = FastAPI(title="HazardLoop", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image."})

    person_results = run_inspection(frame)
    decision = decide(person_results)

    result = {
        "people": person_results,
        "decision": decision,
    }

    s3_key = log_inspection(image_bytes, result)
    result["s3_key"] = s3_key

    return result
