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

from .subspace_routing import DEFAULT_KEYWORD_BOOST, DEFAULT_SUBSPACE_MAP, L2_SUBSPACES

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None
_CACHE_AT: float = 0.0
_TTL_S = 60.0
_FALLBACK_DIR = Path(__file__).resolve().parent / "fallback_packs"
_FALLBACK_PROMPTS_PATH = _FALLBACK_DIR / "prompts_orchestrator.yaml"
_FALLBACK_CANNED_PATH = _FALLBACK_DIR / "canned.yaml"

_FALLBACK_SUBSPACE_MAP = dict(DEFAULT_SUBSPACE_MAP)
_FALLBACK_BOOST = dict(DEFAULT_KEYWORD_BOOST)
_FALLBACK_SUBSPACE_IDS = tuple(L2_SUBSPACES)


def _load_vendored_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        log.error("Vendored pack missing: %s", path)
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.exception("Failed to load vendored pack %s: %s", path, exc)
        return {}


def _load_vendored_prompts() -> dict[str, str]:
    """Load prompt templates from the package YAML (offline / KB-down path)."""
    data = _load_vendored_yaml(_FALLBACK_PROMPTS_PATH)
    prompts = data.get("prompts") or {}
    if not isinstance(prompts, dict):
        return {}
    return {str(k): str(v) for k, v in prompts.items() if v is not None}


def _load_vendored_canned() -> dict[str, Any]:
    return _load_vendored_yaml(_FALLBACK_CANNED_PATH)


def _fallback_payload() -> dict[str, Any]:
    canned = _load_vendored_canned()
    return {
        "pins": {
            "prompts_orchestrator": "fallback-yaml",
            "canned": str(canned.get("version") or "fallback-yaml"),
        },
        "manifest": None,
        "taxonomy": {
            "version": None,
            "subspaces": list(_FALLBACK_SUBSPACE_MAP.keys()),
            "subspace_map": dict(_FALLBACK_SUBSPACE_MAP),
            "keyword_boost": dict(_FALLBACK_BOOST),
            "ui_filter_groups": [],
        },
        "canned": canned,
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
    return dict(get_registry_payload().get("canned") or _load_vendored_canned())


def greeting_message(kind: str = "greeting") -> str:
    g = (canned().get("greetings") or {})
    fb = (_load_vendored_canned().get("greetings") or {})
    return str(g.get(kind) or g.get("greeting") or fb.get("greeting") or "")


def refusal_message(kind: str = "no_evidence") -> str:
    r = (canned().get("refusals") or {})
    fb = (_load_vendored_canned().get("refusals") or {})
    return str(r.get(kind) or fb.get(kind) or fb.get("no_evidence") or "")


def abstention_phrase() -> str:
    r = (canned().get("refusals") or {})
    fb = (_load_vendored_canned().get("refusals") or {})
    return str(r.get("abstention_phrase") or fb.get("abstention_phrase") or "")


def canned_message(key: str, **fmt: Any) -> str:
    """User-facing message from canned.messages (or refusals)."""
    pack = canned()
    msgs = dict(pack.get("messages") or {})
    refusals = dict(pack.get("refusals") or {})
    text = msgs.get(key) or refusals.get(key)
    if text is None:
        fb = _load_vendored_canned()
        text = (fb.get("messages") or {}).get(key) or (fb.get("refusals") or {}).get(key)
    if text is None:
        raise RuntimeError(f"Canned message {key!r} missing from registry")
    if not fmt:
        return str(text)
    try:
        return str(text).format(**fmt)
    except Exception:
        return str(text)


def clarify_option_labels() -> list[str]:
    labels = canned().get("clarify_option_labels")
    if isinstance(labels, list) and labels:
        return [str(x) for x in labels]
    fb = _load_vendored_canned().get("clarify_option_labels") or []
    return [str(x) for x in fb]


def faq_shortlist() -> dict[str, str]:
    """Curated FAQ Q→A map from canned pack."""
    items = canned().get("faq_shortlist")
    if not isinstance(items, list):
        items = (_load_vendored_canned().get("faq_shortlist") or [])
    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q") or "").strip()
        a = str(item.get("a") or "").strip()
        if q and a:
            out[q] = a
    return out


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
