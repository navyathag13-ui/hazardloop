"""Rescore a results file after the hand check in bench/audit_<set>.json.

Photos where the check found a bare head the labels skipped are counted as
violations for every method. Only the vision LLM's disagreements were checked,
so OpenCV false alarms on other photos are left as they were.

Run:  python bench/audited_scores.py --set heldout
"""
import argparse
import json
from pathlib import Path

from run_eval import prf

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="heldout")
    args = ap.parse_args()
    rows = json.loads((ROOT / "bench" / f"results_{args.set}.json").read_text())["rows"]
    audit = json.loads((ROOT / "bench" / f"audit_{args.set}.json").read_text())["checked"]
    missed = {a["photo"] for a in audit if a["verdict"] == "label_missed"}
    fixed = [{**r, "truth": r["truth"] or r["photo"] in missed} for r in rows]
    out = {"set": args.set, "labels_corrected": len(missed), "checked": len(audit),
           "photos_with_violation": sum(r["truth"] for r in fixed)}
    for m in ("opencv", "vision_llm", "cross_check"):
        out[m] = prf(fixed, m)
    out["cross_check"]["violations_missed"] = sum(1 for r in fixed if r["truth"] and r["cross_check_decision"] == "pass")
    (ROOT / "bench" / f"audited_{args.set}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
