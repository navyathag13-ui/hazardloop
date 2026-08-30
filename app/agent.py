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
