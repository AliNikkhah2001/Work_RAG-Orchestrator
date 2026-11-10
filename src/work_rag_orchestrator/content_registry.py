"""Load versioned taxonomy / canned / prompts from KB Manager registry.

Falls back to local defaults when the registry is unreachable so the
orchestrator still boots offline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT: float = 0.0
_TTL_S = 60.0

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


def _fallback_payload() -> dict[str, Any]:
    return {
        "pins": {},
        "manifest": None,
        "taxonomy": {
            "version": None,
            "subspaces": list(_FALLBACK_SUBSPACE_MAP.keys()),
            "subspace_map": dict(_FALLBACK_SUBSPACE_MAP),
            "keyword_boost": dict(_FALLBACK_BOOST),
            "ui_filter_groups": [],
        },
        "canned": dict(_FALLBACK_CANNED),
        "prompts": {"prompts": {}},
        "source": "fallback",
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
    settings = get_settings()
    url = settings.kb_base_url.rstrip("/") + "/registry/api/active"
    try:
        with httpx.Client(timeout=8.0, trust_env=False) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            data["source"] = "kb-registry"
            with _LOCK:
                _CACHE = data
                _CACHE_AT = time.monotonic()
            return data
    except Exception as exc:
        log.warning("KB registry unavailable (%s); using fallback content packs", exc)
        payload = _fallback_payload()
        with _LOCK:
            if _CACHE is None:
                _CACHE = payload
                _CACHE_AT = time.monotonic()
            return _CACHE


def get_registry_payload() -> dict[str, Any]:
    return refresh_registry(force=False)


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
    prompts = ((get_registry_payload().get("prompts") or {}).get("prompts") or {})
    text = prompts.get(name)
    if not text:
        return None
    try:
        return str(text).format(**fmt)
    except Exception:
        return str(text)
