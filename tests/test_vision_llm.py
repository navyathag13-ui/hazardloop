import json
from types import SimpleNamespace

from app import vision_llm as v


def reply(**kw):
    body = {"with_hardhat": 2, "without_hardhat": 1, "unclear_heads": 0, "needs_human_review": False}
    body.update(kw)
    return json.dumps(body)


def scripted(*replies):
    seen = []

    def call(feedback):
        seen.append(feedback)
        return replies[len(seen) - 1]
    return call, seen


def test_valid_reply():
    c = v.parse(reply())
    assert (c.with_hardhat, c.without_hardhat) == (2, 1) and not c.needs_human_review


def test_bad_values_fail():
    for bad in ({"with_hardhat": -1}, {"without_hardhat": 51}, {"extra": 1}, {"unclear_heads": "two"}):
        try:
            v.parse(reply(**bad))
        except Exception:
            continue
        raise AssertionError(f"{bad} should fail")


def test_unclear_heads_force_review():
    assert v.parse(reply(unclear_heads=2)).needs_human_review


def test_retry_sends_errors_back():
    call, seen = scripted(reply(with_hardhat=-3), reply())
    c, meta = v.generate(call)
    assert meta.validated and meta.attempts == 2 and "with_hardhat" in seen[1]


def test_two_bad_replies_fall_back():
    c, meta = v.generate(scripted("not json", reply(without_hardhat=99))[0])
    assert c is None and meta.fallback_used and meta.errors


def test_no_model_configured(monkeypatch):
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(k, raising=False)
    c, meta = v.count_ppe(b"x")
    assert c is None and not meta.validated and meta.attempts == 0


def test_count_with_given_client_sends_the_image():
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply()))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c, meta = v.count_ppe(b"\xff\xd8jpeg", client=client, model="m")
    assert meta.validated and c.without_hardhat == 1
    url = sent["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and sent["response_format"]["json_schema"]["strict"]


def test_to_people_matches_counts():
    people = v.to_people(v.parse(reply()))
    assert sum(not p["ppe_compliant"] for p in people) == 1 and len(people) == 3


def test_refused_request_goes_to_review():
    import httpx
    from openai import BadRequestError

    def create(**kw):
        req = httpx.Request("POST", "https://x")
        raise BadRequestError("refused", response=httpx.Response(400, request=req),
                              body={"code": "content_policy_violation"})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c, meta = v.count_ppe(b"img", client=client, model="m")
    assert c is None and meta.fallback_used and not meta.validated and meta.errors
