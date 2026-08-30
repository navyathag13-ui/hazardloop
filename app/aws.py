"""AWS integration: logs each inspection (image + decision) to S3.

Reads config from environment variables so the app runs locally without AWS
configured (S3 logging is skipped rather than failing the request).
"""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

_BUCKET = os.environ.get("HAZARDLOOP_S3_BUCKET")
_s3 = boto3.client("s3") if _BUCKET else None


def log_inspection(image_bytes: bytes, result: dict) -> str | None:
    """Uploads the source image and JSON result to S3. Returns the S3 key prefix, or None if unconfigured."""
    if not _s3:
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    key_prefix = f"inspections/{ts}-{uuid.uuid4().hex[:8]}"

    try:
        _s3.upload_fileobj(io.BytesIO(image_bytes), _BUCKET, f"{key_prefix}/frame.jpg")
        _s3.put_object(
            Bucket=_BUCKET,
            Key=f"{key_prefix}/result.json",
            Body=json.dumps(result).encode("utf-8"),
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError) as e:
        # Don't fail the inspection just because logging failed.
        print(f"[aws] S3 log failed: {e}")
        return None

    return key_prefix
