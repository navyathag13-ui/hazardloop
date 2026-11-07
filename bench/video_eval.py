"""Run HazardLoop on short worksite clips and measure what smoothing over time does.

Each clip in data/videos/ is sampled at 1 frame per second (the frames written
by the sampling step in README) and every frame goes to the vision LLM. Two
alert rules are compared:

  per_frame  alert on any frame where someone is reported without a hard hat
  smoothed   alert only when that is reported in 2 of the last 3 frames
             (app.video.SmoothedAlerts)

Clip labels are in bench/video_labels.json. On compliant clips every alert is
a false alarm; on the violation clip we want an alert, and how soon.

Run:  python bench/video_eval.py --prompt v2
"""
import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from dotenv import load_dotenv  # noqa: E402

from app import vision_llm  # noqa: E402
from app.video import SmoothedAlerts  # noqa: E402
from run_eval import llm_reading  # noqa: E402

FRAMES = ROOT / "data" / "videos" / "frames"


def events(flags):
    """Number of separate alert runs (False->True switches)."""
    return sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=list(vision_llm.PROMPTS), default="v2")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client, model = vision_llm.get_client()
    labels = json.loads((ROOT / "bench" / "video_labels.json").read_text())["clips"]
    frames = sorted(FRAMES.glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(lambda f: llm_reading("video__" + f.name, f.read_bytes(), client, model, args.prompt), frames))

    clips = {}
    for clip, lab in labels.items():
        readings = [llm_reading("video__" + f.name, f.read_bytes(), client, model, args.prompt)
                    for f in sorted(FRAMES.glob(f"{clip}_*.jpg"))]
        per_frame, unclear = [], []
        for r in readings:
            c = r["counts"]
            per_frame.append(bool(c and c["without_hardhat"] > 0))
            unclear.append(bool(c is None or c["unclear_heads"] > 0))
        sm = SmoothedAlerts()
        smoothed = [sm.update(f) for f in per_frame]
        first = lambda xs: next((i for i, x in enumerate(xs) if x), None)  # noqa: E731
        clips[clip] = {
            "label": lab["label"], "frames": len(readings),
            "per_frame_alert_frames": sum(per_frame), "per_frame_alert_events": events(per_frame),
            "smoothed_alert_frames": sum(smoothed), "smoothed_alert_events": events(smoothed),
            "first_alert_second_per_frame": first(per_frame), "first_alert_second_smoothed": first(smoothed),
            "frames_with_unclear_heads": sum(unclear),
            "per_frame": per_frame, "smoothed": smoothed,
        }

    def total(label, key):
        return sum(c[key] for c in clips.values() if c["label"] == label)

    summary = {"prompt": args.prompt, "model": model}
    for label in ("compliant", "violation", "hard"):
        summary[label] = {
            "clips": sum(1 for c in clips.values() if c["label"] == label),
            "frames": total(label, "frames"),
            "per_frame_alert_frames": total(label, "per_frame_alert_frames"),
            "per_frame_alert_events": total(label, "per_frame_alert_events"),
            "smoothed_alert_frames": total(label, "smoothed_alert_frames"),
            "smoothed_alert_events": total(label, "smoothed_alert_events"),
            "frames_with_unclear_heads": total(label, "frames_with_unclear_heads"),
        }
    secs = [llm_reading("video__" + f.name, f.read_bytes(), client, model, args.prompt)["seconds"] for f in frames]
    summary["median_seconds_per_frame"] = statistics.median(secs)
    out = ROOT / "bench" / f"video_results_{args.prompt}.json"
    out.write_text(json.dumps({"summary": summary, "clips": clips}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for k, c in clips.items():
        print(k, c["label"], "per-frame", "".join("x" if f else "." for f in c["per_frame"]),
              "smoothed", "".join("x" if f else "." for f in c["smoothed"]))


if __name__ == "__main__":
    main()
