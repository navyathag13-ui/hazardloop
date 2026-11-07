"""Score HazardLoop on labeled worksite photos.

For each photo in a set from bench/sets.json it runs:
  opencv      the original stage: YuNet face detector + hard-hat color check
  vision_llm  gpt-4.1-mini (Azure OpenAI) counting heads with and without hard hats
  cross_check the vision LLM decision, sent to human review when it marks
              unclear heads or disagrees with OpenCV (app.agent.cross_check)

and compares each with the true labels. The positive class is "this photo has
at least one person without a hard hat", so precision, recall and F1 are about
catching violations. It also reports decision accuracy against the agent rules
applied to the true counts, head-count error and, for cross_check, how many
photos went to a person and how many violations still slipped through.

Vision LLM replies are cached in bench/cache/ so a rerun costs nothing.

Run:  python bench/run_eval.py --set tuning
      python bench/run_eval.py --set heldout
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from app import vision_llm  # noqa: E402
from app.agent import cross_check, decide  # noqa: E402
from app.detector import run_inspection  # noqa: E402

RAW = ROOT / "data" / "raw" / "test"
CACHE = ROOT / "bench" / "cache"


def prf(rows, key):
    tp = sum(1 for r in rows if r[key] and r["truth"])
    fp = sum(1 for r in rows if r[key] and not r["truth"])
    fn = sum(1 for r in rows if not r[key] and r["truth"])
    tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3)}


def llm_reading(name: str, image_bytes: bytes, client, model: str) -> dict:
    path = CACHE / f"{model}__{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    start = time.perf_counter()
    counts, meta = vision_llm.count_ppe(image_bytes, client=client, model=model)
    out = {"counts": counts.model_dump() if counts else None, "meta": meta.model_dump(),
           "seconds": round(time.perf_counter() - start, 2)}
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def run(set_name: str) -> dict:
    load_dotenv(ROOT / ".env")
    client, model = vision_llm.get_client()
    if client is None:
        sys.exit("Azure OpenAI is not configured in .env.")
    photos = json.loads((ROOT / "bench" / "sets.json").read_text())[set_name]
    with ThreadPoolExecutor(max_workers=1) as pool:  # one at a time: the Azure deployment has a low request quota
        list(pool.map(lambda n: llm_reading(n, (RAW / n).read_bytes(), client, model), photos))
    rows = []
    for name, gold in photos.items():
        image_bytes = (RAW / name).read_bytes()
        frame = cv2.imread(str(RAW / name))
        cv_people = run_inspection(frame)
        cv_dec = decide(cv_people)
        reading = llm_reading(name, image_bytes, client, model)
        counts = vision_llm.PPECounts(**reading["counts"]) if reading["counts"] else None
        llm_dec = decide(vision_llm.to_people(counts)) if counts else {"action": "pass", "violations": 0}
        cc = cross_check(cv_people, counts, reading["meta"]["validated"])
        rows.append({
            "photo": name,
            "truth": gold["no_hardhat"] > 0,
            "true_heads": gold["hardhat"] + gold["no_hardhat"],
            "true_decision": gold["decision"],
            "opencv": cv_dec["violations"] > 0,
            "opencv_heads": len(cv_people),
            "opencv_decision": cv_dec["action"],
            "vision_llm": bool(counts and counts.without_hardhat > 0),
            "vision_llm_heads": (counts.with_hardhat + counts.without_hardhat + counts.unclear_heads) if counts else 0,
            "vision_llm_decision": llm_dec["action"],
            "cross_check": cc["action"] != "pass",
            "cross_check_decision": cc["action"],
            "cross_check_review": cc["needs_human_review"],
            "llm_attempts": reading["meta"]["attempts"],
            "llm_fallback": reading["meta"]["fallback_used"],
            "llm_seconds": reading["seconds"],
        })

    n = len(rows)
    summary = {"set": set_name, "photos": n, "photos_with_violation": sum(r["truth"] for r in rows), "model": model}
    for m in ("opencv", "vision_llm"):
        summary[m] = {
            **prf(rows, m),
            "decision_accuracy": f"{sum(r[m + '_decision'] == r['true_decision'] for r in rows)}/{n}",
            "head_count_mae": round(statistics.mean(abs(r[m + "_heads"] - r["true_heads"]) for r in rows), 2),
        }
    reviewed = [r for r in rows if r["cross_check_review"]]
    summary["cross_check"] = {
        **prf(rows, "cross_check"),
        "sent_to_review": f"{len(reviewed)}/{n}",
        "violations_missed": sum(1 for r in rows if r["truth"] and r["cross_check_decision"] == "pass"),
        "llm_retries": sum(1 for r in rows if r["llm_attempts"] > 1),
        "llm_fallbacks": sum(1 for r in rows if r["llm_fallback"]),
        "llm_median_seconds": statistics.median(r["llm_seconds"] for r in rows),
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["tuning", "heldout"], default="tuning")
    args = ap.parse_args()
    result = run(args.set)
    out = ROOT / "bench" / f"results_{args.set}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
