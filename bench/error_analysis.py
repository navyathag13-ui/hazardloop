"""Where does the vision LLM go wrong? Breaks a results file down with pandas.

Joins each photo's result with its dataset annotations and groups by
  crowd  number of labeled heads in the photo
  size   side length in pixels of the smallest labeled head (sqrt of box area,
         640x640 photos); photos with no labeled heads are left out of this one

and reports precision and the false-alarm rate (false alarms / photos without a
violation) per group. Head size comes from the labels, so it describes the
footage, not something the system knows at run time: the point is to say what
camera setup the model can be trusted with.

Run:  python bench/error_analysis.py --set heldout2 --prompt v2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "test"
CROWD_BINS, CROWD_LABELS = [-1, 1, 3, 6, 100], ["0-1", "2-3", "4-6", "7+"]
SIZE_BINS, SIZE_LABELS = [0, 24, 48, 96, 1000], ["<24px", "24-48px", "48-96px", "96px+"]


def smallest_head_px(coco: dict) -> dict:
    ids = {im["id"]: im["file_name"] for im in coco["images"]}
    areas = defaultdict(list)
    for a in coco["annotations"]:
        areas[ids[a["image_id"]]].append(a["bbox"][2] * a["bbox"][3])
    return {name: min(v) ** 0.5 for name, v in areas.items()}


def breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.assign(tp=df.vision_llm & df.truth, fp=df.vision_llm & ~df.truth, fn=~df.vision_llm & df.truth)
    g = df.groupby(col, observed=True).agg(
        photos=("photo", "size"), violations=("truth", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
    g["precision"] = (g.tp / (g.tp + g.fp)).round(2)
    g["false_alarm_rate"] = (g.fp / (g.photos - g.violations)).round(2)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout2")
    ap.add_argument("--prompt", default="v2")
    args = ap.parse_args()
    suffix = "" if args.prompt == "v1" else f"_{args.prompt}"
    rows = json.loads((ROOT / "bench" / f"results_{args.set}{suffix}.json").read_text())["rows"]
    df = pd.DataFrame(rows)
    sizes = smallest_head_px(json.loads((RAW / "_annotations.coco.json").read_text()))
    df["smallest_head_px"] = df.photo.map(sizes)
    df["crowd"] = pd.cut(df.true_heads, CROWD_BINS, labels=CROWD_LABELS)
    df["size"] = pd.cut(df.smallest_head_px, SIZE_BINS, labels=SIZE_LABELS)
    out = {}
    for col in ("crowd", "size"):
        g = breakdown(df, col)
        print(f"\nby {col}\n{g}")
        out[col] = g.reset_index().astype({col: str}).to_dict(orient="records")
    path = ROOT / "bench" / f"error_analysis_{args.set}{suffix}.json"
    path.write_text(json.dumps(out, indent=1, default=int) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
