"""Agent decision stage: reasons over OpenCV detection results and decides an action.

This is the perception -> decision -> action loop the Agentic Vision track
judges on: vision output here changes what the system does next, not just
what it reports.
"""


def decide(person_results: list[dict]) -> dict:
    """Turns per-person PPE results into a site-level decision.

    Actions:
      - "pass": no personnel, or everyone compliant
      - "flag": some violations, below escalation threshold -> route to human review
      - "escalate": violation rate/count crosses threshold -> immediate escalation
    """
    total = len(person_results)
    violations = [r for r in person_results if not r["ppe_compliant"]]
    violation_count = len(violations)

    if total == 0:
        return {
            "action": "pass",
            "reason": "No personnel detected in frame.",
            "people_detected": 0,
            "violations": 0,
        }

    violation_rate = violation_count / total

    if violation_count == 0:
        action = "pass"
        reason = f"All {total} detected personnel are PPE-compliant."
    elif violation_count >= 3 or violation_rate >= 0.5:
        action = "escalate"
        reason = (
            f"{violation_count}/{total} personnel non-compliant "
            f"({violation_rate:.0%}) -- exceeds escalation threshold."
        )
    else:
        action = "flag"
        reason = f"{violation_count}/{total} personnel non-compliant -- routed to human review."

    return {
        "action": action,
        "reason": reason,
        "people_detected": total,
        "violations": violation_count,
    }


def cross_check(cv_people: list[dict], llm_counts, llm_validated: bool) -> dict:
    """Decision from the vision LLM, cross-checked against the OpenCV stage.

    The LLM's counts drive the decision. A person reviews the photo (the
    action is at least "flag") when:
      - the LLM reply failed validation, so only the OpenCV result is left
      - the LLM marked heads it could not judge
      - OpenCV sees someone without a hard hat but the LLM does not

    Only that direction of disagreement goes to review: when the LLM reports
    a violation the decision already flags or escalates it, so a second
    opinion there adds review work without catching anything new.
    """
    cv_violation = any(not p["ppe_compliant"] for p in cv_people)
    if not llm_validated or llm_counts is None:
        base = decide(cv_people)
        return {**base, "action": "flag" if base["action"] == "pass" else base["action"],
                "needs_human_review": True, "source": "opencv",
                "reason": "Vision model reply failed validation; OpenCV result sent for review. " + base["reason"]}

    people = ([{"ppe_compliant": True}] * llm_counts.with_hardhat
              + [{"ppe_compliant": False}] * llm_counts.without_hardhat)
    base = decide(people)
    llm_violation = llm_counts.without_hardhat > 0
    reasons = []
    if llm_counts.unclear_heads:
        reasons.append(f"{llm_counts.unclear_heads} head(s) could not be judged")
    if cv_violation and not llm_violation:
        reasons.append("OpenCV saw a missing hard hat that the vision model did not")
    if not reasons:
        return {**base, "needs_human_review": False, "source": "vision_llm"}
    action = "flag" if base["action"] == "pass" else base["action"]
    return {**base, "action": action, "needs_human_review": True, "source": "vision_llm",
            "reason": base["reason"] + " Review: " + "; ".join(reasons) + "."}
