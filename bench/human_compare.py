"""Compare a person's photo labels with the dataset labels and with the vision LLM.

bench/human/photos.json lists 50 photos from the heldout2 set (20 the dataset
marks as having a violation, 30 it marks as compliant, drawn with a fixed
seed). bench/human/labels_navyatha.json holds Navyatha's answers from
bench/human/index.html: violation, all_hardhat or unclear, given without seeing
the dataset labels or the model's answers.

Run:  python bench/human_compare.py --prompt v2
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def agree(a, b, photos):
    both = [p for p in photos if a.get(p) is not None and b.get(p) is not None]
    return sum(a[p] == b[p] for p in both), len(both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v2")
    args = ap.parse_args()
    photos = json.loads((ROOT / "bench" / "human" / "photos.json").read_text())
    human_raw = json.loads((ROOT / "bench" / "human" / "labels_navyatha.json").read_text())
    sets = json.loads((ROOT / "bench" / "sets.json").read_text())["heldout2"]
    dataset = {p: sets[p]["no_hardhat"] > 0 for p in photos}
    human = {p: (None if v == "unclear" else v == "violation") for p, v in human_raw.items()}
    out = {"photos": len(photos), "human_unclear": sum(v is None for v in human.values())}
    a, n = agree(human, dataset, photos)
    out["human_vs_dataset"] = f"{a}/{n}"
    out["human_violation_dataset_compliant"] = sum(1 for p in photos if human.get(p) is True and not dataset[p])
    out["human_compliant_dataset_violation"] = sum(1 for p in photos if human.get(p) is False and dataset[p])
    suffix = "" if args.prompt == "v1" else f"_{args.prompt}"
    res = ROOT / "bench" / f"results_heldout2{suffix}.json"
    if res.exists():
        rows = {r["photo"]: r for r in json.loads(res.read_text())["rows"]}
        model = {p: rows[p]["vision_llm"] for p in photos if p in rows}
        a, n = agree(model, human, photos)
        out[f"model_{args.prompt}_vs_human"] = f"{a}/{n}"
        a, n = agree(model, dataset, photos)
        out[f"model_{args.prompt}_vs_dataset"] = f"{a}/{n}"
        out["model_violation_human_compliant"] = sum(1 for p in model if model[p] and human.get(p) is False)
        out["model_compliant_human_violation"] = sum(1 for p in model if not model[p] and human.get(p) is True)
    (ROOT / "bench" / f"human_compare{suffix}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
