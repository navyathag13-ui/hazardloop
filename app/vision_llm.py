"""Vision LLM stage: counts workers with and without a hard hat in a photo.

The image goes to a vision-capable chat model (gpt-4.1-mini on Azure OpenAI)
with a strict JSON schema, so the reply always has the same shape. A strict
schema only guarantees shape, so the reply is then validated:

  - counts are whole numbers from 0 to 50
  - unclear_heads (heads too small, blurred or turned away to judge) is
    reported separately instead of being forced into a guess
  - needs_human_review is forced on when any head is unclear

A reply that fails validation is sent back once with the errors listed. If
the retry fails too, the caller gets no counts and needs_human_review=true, so
a bad reading is never passed off as a good one.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_RETRIES = 1


class PPECounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    with_hardhat: int = Field(..., ge=0, le=50)
    without_hardhat: int = Field(..., ge=0, le=50)
    unclear_heads: int = Field(..., ge=0, le=50)
    needs_human_review: bool


class VisionMeta(BaseModel):
    validated: bool
    attempts: int
    fallback_used: bool = False
    errors: list[str] = Field(default_factory=list)


SCHEMA: dict = {
    "type": "object",
    "properties": {
        "with_hardhat": {"type": "integer", "description": "Visible people whose head is covered by a hard hat or safety helmet."},
        "without_hardhat": {"type": "integer", "description": "Visible people whose head is clearly uncovered or covered by something that is not a hard hat (cap, hood, hair)."},
        "unclear_heads": {"type": "integer", "description": "Heads too small, blurred, cut off or turned away to judge."},
        "needs_human_review": {"type": "boolean", "description": "True if the photo is hard to judge."},
    },
    "required": ["with_hardhat", "without_hardhat", "unclear_heads", "needs_human_review"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {"name": "ppe_counts", "strict": True, "schema": SCHEMA},
}

SYSTEM_PROMPT = """You check worksite photos for hard-hat compliance.
Count every visible person's head, including small people in the background.
A baseball cap, beanie, hood or bare head is NOT a hard hat.
If you cannot tell whether a head has a hard hat, count it as unclear instead of guessing.
Reply with a JSON object matching the schema."""


def build_messages(image_bytes: bytes, mime: str = "image/jpeg") -> list[dict]:
    b64 = base64.b64encode(image_bytes).decode()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Count the workers with and without hard hats in this photo."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]},
    ]


def parse(raw: str) -> PPECounts:
    counts = PPECounts.model_validate(json.loads(raw))
    if counts.unclear_heads:
        counts.needs_human_review = True
    return counts


def _errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors()]
    return [f"{type(exc).__name__}: {exc}"]


def _feedback(errors: list[str]) -> str:
    return "Your previous reply failed validation:\n- " + "\n- ".join(errors) + "\nReply again with only the JSON object."


def generate(call: Callable[[Optional[str]], str], max_retries: int = MAX_RETRIES) -> tuple[Optional[PPECounts], VisionMeta]:
    """call(None) gives the first reply; call(feedback) a retry after the model sees its errors."""
    errors: list[str] = []
    for attempt in range(1, max_retries + 2):
        raw = call(_feedback(errors) if attempt > 1 else None)
        try:
            return parse(raw), VisionMeta(validated=True, attempts=attempt)
        except (ValueError, ValidationError) as exc:
            errors = _errors(exc)
    return None, VisionMeta(validated=False, attempts=max_retries + 1, fallback_used=True, errors=errors)


def get_client():
    """Azure OpenAI client from the environment, or (None, "") when it is not configured."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not (endpoint and key and deployment):
        return None, ""
    from openai import AzureOpenAI

    version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version, max_retries=8), deployment


def count_ppe(image_bytes: bytes, client=None, model: str = "") -> tuple[Optional[PPECounts], VisionMeta]:
    if client is None:
        client, model = get_client()
    if client is None:
        return None, VisionMeta(validated=False, attempts=0, errors=["No vision model is configured."])
    messages = build_messages(image_bytes)

    def call(feedback: Optional[str]) -> str:
        msgs = list(messages)
        if feedback:
            msgs.append({"role": "user", "content": feedback})
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0, response_format=RESPONSE_FORMAT)
        return r.choices[0].message.content or ""

    try:
        return generate(call)
    except Exception as exc:  # refused (e.g. content filter), rate limited after retries, network
        from openai import APIError

        if not isinstance(exc, APIError):
            raise
        return None, VisionMeta(validated=False, attempts=1, fallback_used=True,
                                errors=[f"{type(exc).__name__}: {getattr(exc, 'code', None) or exc}"])


def to_people(counts: PPECounts) -> list[dict]:
    """Per-person results in the same shape the OpenCV stage returns, for the agent."""
    return ([{"bbox": None, "ppe_compliant": True}] * counts.with_hardhat
            + [{"bbox": None, "ppe_compliant": False}] * counts.without_hardhat)
