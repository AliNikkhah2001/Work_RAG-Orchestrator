"""Load versioned taxonomy / canned / prompts from KB Manager registry.

Prompt *text* is never hardcoded in graph nodes. Nodes call
:func:`require_prompt` which:

1. Prefers live ``GET {kb}/registry/api/active`` (source=``kb-registry``)
2. Falls back to the vendored YAML under ``fallback_packs/`` when KB is down
   (source=``fallback-yaml``) so offline boot still works without string
   literals scattered across the codebase.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT: float = 0.0
_TTL_S = 60.0
_FALLBACK_PROMPTS_PATH = Path(__file__).resolve().parent / "fallback_packs" / "prompts_orchestrator.yaml"

_FALLBACK_SUBSPACE_MAP = {
    "cheque": "گزارش اعتبارسنجی چک",
    "facilities_person": "اشخاص حقیقی/اعتبارسنجی تسهیلات",
    "facilities_legal": "اشخاص حقوقی/اعتبارسنجی تسهیلات",
    "general": "پایگاه دانش عمومی",
}
_FALLBACK_BOOST = {
    "cheque": 3.5,
    "facilities_person": 3.0,
    "facilities_legal": 3.0,
    "general": 2.5,
}
_FALLBACK_CANNED = {
    "greetings": {
        "greeting": "سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتباری، امتیاز و رتبه، یا سوابق و چک‌ها بفرمایید.",
        "farewell": "خداحافظ! اگر سوال دیگری داشتید در خدمتم.",
        "thanks": "خواهش می‌کنم. اگر سوال دیگری دارید بفرمایید.",
    },
    "refusals": {
        "no_evidence": "بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.",
        "weak_evidence": "بر اساس منابع بازیابی‌شده، پاسخ مستقیم در متن موجود نیست؛ لطفاً سوال را دقیق‌تر بپرسید.",
        "abstention_phrase": "پاسخی برای این سوال یافت نشد",
        "blocked_default": "I cannot comply with that request.",
    },
    "greeting_triggers": ["سلام", "درود", "hello", "hi", "hey", "صبح بخیر", "وقت بخیر"],
}


def _load_vendored_prompts() -> dict[str, str]:
    """Load prompt templates from the package YAML (offline / KB-down path)."""
    if not _FALLBACK_PROMPTS_PATH.is_file():
        log.error("Vendored prompts pack missing: %s", _FALLBACK_PROMPTS_PATH)
        return {}
    try:
        data = yaml.safe_load(_FALLBACK_PROMPTS_PATH.read_text(encoding="utf-8")) or {}
        prompts = data.get("prompts") or {}
        if not isinstance(prompts, dict):
            return {}
        return {str(k): str(v) for k, v in prompts.items() if v is not None}
    except Exception as exc:
        log.exception("Failed to load vendored prompts pack: %s", exc)
        return {}


def _fallback_payload() -> dict[str, Any]:
    return {
        "pins": {"prompts_orchestrator": "fallback-yaml"},
        "manifest": None,
        "taxonomy": {
            "version": None,
            "subspaces": list(_FALLBACK_SUBSPACE_MAP.keys()),
            "subspace_map": dict(_FALLBACK_SUBSPACE_MAP),
            "keyword_boost": dict(_FALLBACK_BOOST),
            "ui_filter_groups": [],
        },
        "canned": dict(_FALLBACK_CANNED),
        "prompts": {"prompts": _load_vendored_prompts(), "version": "fallback-yaml"},
        "source": "fallback-yaml",
    }


def refresh_registry(force: bool = False) -> dict[str, Any]:
    """Fetch ``GET {kb}/registry/api/active`` and cache it."""
    global _CACHE, _CACHE_AT
    with _LOCK:
        if (
            not force
            and _CACHE is not None
            and (time.monotonic() - _CACHE_AT) < _TTL_S
        ):
            return _CACHE
    # Read KB URL from env — do NOT call get_settings() here.
    # get_settings() overlays taxonomy from this registry and would recurse.
    kb_base = (os.getenv("KB_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    url = kb_base + "/registry/api/active"
    try:
        with httpx.Client(timeout=8.0, trust_env=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            data["source"] = "kb-registry"
            prompts_obj = data.get("prompts") or {}
            prompt_map = prompts_obj.get("prompts") if isinstance(prompts_obj, dict) else {}
            if not prompt_map:
                raise RuntimeError("KB registry returned empty prompts map")
            with _LOCK:
                _CACHE = data
                _CACHE_AT = time.monotonic()
            log.info(
                "Loaded content registry from KB (pins=%s, prompt_keys=%d)",
                data.get("pins"),
                len(prompt_map),
            )
            return data
    except Exception as exc:
        log.warning("KB registry unavailable (%s); using vendored prompt YAML", exc)
        payload = _fallback_payload()
        with _LOCK:
            # Prefer keeping a previous successful KB cache if present
            if _CACHE is not None and (_CACHE.get("source") == "kb-registry"):
                return _CACHE
            _CACHE = payload
            _CACHE_AT = time.monotonic()
            return _CACHE


def get_registry_payload() -> dict[str, Any]:
    return refresh_registry(force=False)


def registry_source() -> str:
    return str(get_registry_payload().get("source") or "unknown")


def prompt_catalog() -> dict[str, str]:
    prompts = ((get_registry_payload().get("prompts") or {}).get("prompts") or {})
    return {str(k): str(v) for k, v in prompts.items()}


def subspace_ids() -> tuple[str, ...]:
    tax = get_registry_payload().get("taxonomy") or {}
    ids = tax.get("subspaces") or list(_FALLBACK_SUBSPACE_MAP.keys())
    return tuple(ids)


def subspace_map() -> dict[str, str]:
    tax = get_registry_payload().get("taxonomy") or {}
    return dict(tax.get("subspace_map") or _FALLBACK_SUBSPACE_MAP)


def keyword_boost_map() -> dict[str, float]:
    tax = get_registry_payload().get("taxonomy") or {}
    raw = tax.get("keyword_boost") or _FALLBACK_BOOST
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out or dict(_FALLBACK_BOOST)


def canned() -> dict[str, Any]:
    return dict(get_registry_payload().get("canned") or _FALLBACK_CANNED)


def greeting_message(kind: str = "greeting") -> str:
    g = (canned().get("greetings") or {})
    return str(g.get(kind) or g.get("greeting") or _FALLBACK_CANNED["greetings"]["greeting"])


def refusal_message(kind: str = "no_evidence") -> str:
    r = (canned().get("refusals") or {})
    return str(r.get(kind) or _FALLBACK_CANNED["refusals"]["no_evidence"])


def abstention_phrase() -> str:
    r = (canned().get("refusals") or {})
    return str(r.get("abstention_phrase") or _FALLBACK_CANNED["refusals"]["abstention_phrase"])


def prompt(name: str, **fmt: Any) -> str | None:
    """Return a formatted prompt template, or None if missing."""
    prompts = prompt_catalog()
    text = prompts.get(name)
    if not text:
        return None
    if not fmt:
        return str(text)
    try:
        return str(text).format(**fmt)
    except Exception as exc:
        log.warning("prompt %r format failed (%s); returning raw template", name, exc)
        return str(text)


def require_prompt(name: str, **fmt: Any) -> str:
    """Load prompt *name* from registry (or vendored YAML). Never use node hardcodes.

    Raises ``RuntimeError`` if the template is missing so we fail loud instead of
    silently answering with an embedded string.
    """
    text = prompt(name, **fmt)
    if not text or not str(text).strip():
        raise RuntimeError(
            f"Prompt {name!r} missing from content registry "
            f"(source={registry_source()!r}). "
            "Add it to packs/prompts/orchestrator and pin via retrieval.active_packs."
        )
    return str(text)
