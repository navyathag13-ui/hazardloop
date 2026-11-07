from app.agent import cross_check, decide
from app.vision_llm import PPECounts

OK = {"ppe_compliant": True}
BAD = {"ppe_compliant": False}


def counts(w, wo, unclear=0):
    return PPECounts(with_hardhat=w, without_hardhat=wo, unclear_heads=unclear, needs_human_review=bool(unclear))


def test_decide_thresholds():
    assert decide([])["action"] == "pass"
    assert decide([OK, OK])["action"] == "pass"
    assert decide([OK, OK, OK, BAD])["action"] == "flag"
    assert decide([OK, BAD])["action"] == "escalate"          # 50% rate
    assert decide([OK] * 10 + [BAD] * 3)["action"] == "escalate"  # 3 or more


def test_cross_check_agreement_needs_no_review():
    d = cross_check([OK, OK], counts(2, 0), True)
    assert d["action"] == "pass" and not d["needs_human_review"] and d["source"] == "vision_llm"


def test_llm_decision_drives_the_result():
    d = cross_check([OK], counts(1, 1), True)
    assert d["action"] == "escalate" and not d["needs_human_review"]


def test_opencv_violation_the_llm_missed_goes_to_review():
    d = cross_check([BAD], counts(1, 0), True)
    assert d["action"] == "flag" and d["needs_human_review"]


def test_unclear_heads_go_to_review():
    d = cross_check([], counts(2, 0, unclear=1), True)
    assert d["action"] == "flag" and d["needs_human_review"]


def test_failed_llm_falls_back_to_opencv_with_review():
    d = cross_check([OK], None, False)
    assert d["source"] == "opencv" and d["needs_human_review"] and d["action"] == "flag"
    d = cross_check([BAD], None, False)
    assert d["action"] == "escalate" and d["needs_human_review"]
