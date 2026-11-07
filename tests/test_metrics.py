import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from run_eval import prf  # noqa: E402


def test_precision_recall_f1():
    rows = [{"truth": True, "m": True}, {"truth": True, "m": False},
            {"truth": False, "m": True}, {"truth": False, "m": False}]
    s = prf(rows, "m")
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5


def test_no_predictions_gives_zero_not_an_error():
    s = prf([{"truth": True, "m": False}], "m")
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_breakdown_counts_and_rates():
    import pandas as pd
    from error_analysis import breakdown

    df = pd.DataFrame({
        "photo": list("abcd"), "g": ["x", "x", "y", "y"],
        "truth": [True, False, False, False], "vision_llm": [True, True, False, True],
    })
    g = breakdown(df, "g")
    assert g.loc["x", "tp"] == 1 and g.loc["x", "fp"] == 1 and g.loc["x", "precision"] == 0.5
    assert g.loc["y", "false_alarm_rate"] == 0.5
