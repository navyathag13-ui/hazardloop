"""Draw numbered contact sheets of the photos where a method and the labels disagree,
so each disagreement can be checked by eye and recorded in bench/audit_<set>.json.

Run:  python bench/contact_sheet.py --set heldout --method vision_llm
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout")
    ap.add_argument("--method", default="vision_llm")
    ap.add_argument("--out", default=str(ROOT / "bench" / "audit_sheets"))
    args = ap.parse_args()
    rows = json.loads((ROOT / "bench" / f"results_{args.set}.json").read_text())["rows"]
    wrong = [r for r in rows if r[args.method] != r["truth"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for start in range(0, len(wrong), 8):
        tiles = []
        for i, r in enumerate(wrong[start:start + 8]):
            img = cv2.resize(cv2.imread(str(RAW / r["photo"])), (400, 400))
            kind = "FP" if r[args.method] else "FN"
            cv2.putText(img, f"{start + i + 1} {kind}", (8, 36), 0, 1.2, (0, 0, 255), 3)
            tiles.append(img)
            index.append({"n": start + i + 1, "photo": r["photo"], "kind": kind})
        while len(tiles) < 8:
            tiles.append(np.zeros((400, 400, 3), np.uint8))
        cv2.imwrite(str(out / f"{args.set}_{args.method}_{start // 8 + 1}.jpg"),
                    np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    (out / f"{args.set}_{args.method}_index.json").write_text(json.dumps(index, indent=1))
    print(len(wrong), "disagreements ->", out)


if __name__ == "__main__":
    main()
