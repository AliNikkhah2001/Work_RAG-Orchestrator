"""Langfuse tracing helper — self-hosted. Supports SDK v4 and direct HTTP fallback."""
from __future__ import annotations

import logging
import os
import time
import uuid
import threading
from functools import lru_cache
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Simple HTTP fallback that posts to self-hosted collector at :3000
# Works even when Langfuse SDK version mismatches or Docker is unprivileged
_fallback_host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")


def _auth():
    """Basic auth for real Langfuse (public key = username). Fallback ignores it."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    return (pk, sk) if pk and sk else None


def _now_iso() -> str:
    """UTC ISO-8601 with Z suffix (what Langfuse v2 ingestion expects)."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _envelope(ev_type: str, body: dict) -> dict:
    """Langfuse v2 ingestion envelope: top-level id + ISO timestamp + type + body."""
    return {"id": uuid.uuid4().hex, "timestamp": _now_iso(), "type": ev_type, "body": body}


def _direct_trace(request_id: str, name: str, input_text: str, metadata: dict | None = None):
    """Direct POST to /api/public/ingestion (works for fallback collector and real Langfuse v2)."""
    try:
        host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
        url = f"{host.rstrip('/')}/api/public/ingestion"
        payload = {
            "batch": [
                _envelope(
                    "trace-create",
                    {
                        "id": request_id,
                        "name": name,
                        "input": input_text[:2000],
                        "metadata": metadata or {},
                        "timestamp": _now_iso(),
                    },
                )
            ]
        }
        # Fire-and-forget with short timeout, don't block request
        def _post():
            try:
                httpx.post(url, json=payload, auth=_auth(), timeout=2.0)
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()
    except Exception as e:
        log.warning("Direct trace failed: %s", e)


def _direct_update(request_id: str, output: str, metadata: dict | None = None):
    # NOTE: Langfuse v2 has no "trace-update" event; re-sending trace-create
    # with the same id upserts (input preserved, output set).
    try:
        host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
        url = f"{host.rstrip('/')}/api/public/ingestion"
        payload = {
            "batch": [
                _envelope(
                    "trace-create",
                    {
                        "id": request_id,
                        "output": output[:2000],
                        "metadata": metadata or {},
                        "timestamp": _now_iso(),
                    },
                )
            ]
        }

        def _post():
            try:
                httpx.post(url, json=payload, auth=_auth(), timeout=2.0)
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()
    except Exception:
        pass


# SDK wrapper — try to use Langfuse SDK v4 if available, else fallback
@lru_cache
def get_langfuse():
    if os.getenv("TRACE_ENABLED", "true").lower() in ("0", "false", "no"):
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-mvp-local")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-mvp-local")
        client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        log.info("Langfuse enabled host=%s", host)
        return client
    except Exception as e:
        log.warning("Langfuse not available: %s", e)
        return None


def start_trace(request_id: str, name: str, input_text: str, metadata: dict | None = None):
    """Start trace via SDK or direct HTTP."""
    _direct_trace(request_id, name, input_text, metadata)
    # Also try SDK if present (best effort, don't fail request)
    try:
        lf = get_langfuse()
        if lf is not None:
            # SDK v4 uses start_observation / createTrace, try each
            for method in ("trace", "create_trace", "start_observation", "start_as_current_observation"):
                if hasattr(lf, method):
                    try:
                        getattr(lf, method)(name=name, id=request_id, input=input_text[:1000], metadata=metadata)
                        break
                    except Exception:
                        continue
    except Exception:
        pass


def update_trace(request_id: str, output: str, metadata: dict | None = None):
    _direct_update(request_id, output, metadata)
    try:
        lf = get_langfuse()
        if lf is not None:
            for method in ("score", "flush"):
                pass
            lf.flush()  # type: ignore
    except Exception:
        pass
