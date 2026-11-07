"""Pick the labeled photo sets used by bench/run_eval.py.

Source: the test split of the Hard Hats dataset (Roboflow Universe, CC BY 4.0),
2,001 worksite photos at 640x640 with every head boxed as hardhat or
no-hardhat. Download it to data/raw/test (see README), then run this once.

Two disjoint sets are drawn with a fixed seed:
  tuning   100 photos  the only photos looked at while changing thresholds or prompts
  heldout  300 photos  scored once at the end (prompt v1)
  heldout2 300 photos  drawn after the prompt was revised, scored once with v1 and v2

Each photo's label is the number of heads with and without a hard hat, and the
decision the agent rules give for those true counts (pass, flag or escalate).

Run:  python bench/make_sets.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import decide  # noqa: E402

RAW = ROOT / "data" / "raw" / "test"
SEED = 7
SIZES = {"tuning": 100, "heldout": 300, "heldout2": 300}  # heldout2 added later; earlier sets unchanged


def label_photos() -> dict:
    coco = json.loads((RAW / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = defaultdict(lambda: {"hardhat": 0, "no_hardhat": 0})
    for a in coco["annotations"]:
        counts[a["image_id"]]["hardhat" if names[a["category_id"]] == "hardhat" else "no_hardhat"] += 1
    labels = {}
    for im in coco["images"]:
        c = counts[im["id"]]
        people = [{"ppe_compliant": True}] * c["hardhat"] + [{"ppe_compliant": False}] * c["no_hardhat"]
        labels[im["file_name"]] = {**c, "decision": decide(people)["action"]}
    return labels


def main() -> None:
    labels = label_photos()
    names = sorted(labels)
    random.Random(SEED).shuffle(names)
    out, start = {}, 0
    for set_name, size in SIZES.items():
        out[set_name] = {n: labels[n] for n in names[start:start + size]}
        start += size
    (ROOT / "bench" / "sets.json").write_text(json.dumps(out, indent=1) + "\n")
    for set_name, photos in out.items():
        viol = sum(1 for v in photos.values() if v["no_hardhat"])
        dec = defaultdict(int)
        for v in photos.values():
            dec[v["decision"]] += 1
        print(f"{set_name}: {len(photos)} photos, {viol} with a violation, decisions {dict(dec)}")


if __name__ == "__main__":
    main()
