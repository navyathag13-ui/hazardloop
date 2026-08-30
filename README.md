# HazardLoop

See it, decide, act — AI-powered safety inspection that closes the loop.

HazardLoop is an agentic vision system for site safety inspection. It analyzes
a worksite image, detects personnel and checks PPE (hard-hat) compliance
using OpenCV 5, then an agent reasons over the results and decides whether to
**pass**, **flag** for human review, or **escalate** — closing the loop
between what the camera sees and what happens next. Built for the
[OpenCV AI Competition 2026, powered by AWS](https://opencv-5.devpost.com/)
(Agentic Vision track).

## Architecture

```
image/video --> OpenCV 5 detection (app/detector.py)
                  - person detection (HOG)
                  - PPE/hard-hat color-heuristic check
            --> agent decision (app/agent.py)
                  - pass / flag / escalate
            --> AWS S3 logging (app/aws.py, optional)
            --> FastAPI endpoint (app/main.py)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Then POST an image to `/inspect`:

```bash
curl -X POST -F "file=@samples/example.jpg" http://127.0.0.1:8000/inspect
```

## AWS

Set `HAZARDLOOP_S3_BUCKET` (and standard AWS credentials via env/CLI config)
to enable inspection logging to S3. Without it, the app runs fully locally
and skips AWS logging.

## Status

Early scaffold: person detection + color-heuristic PPE check + rule-based
agent decision. Next: swap the color heuristic for a trained hard-hat
detection model, add video-stream support, and deploy to AWS (Lambda/EC2 +
S3), optionally accelerated with COOL on Graviton.
