"""Ensure orchestrator LLM prompts load from registry / vendored YAML — never node hardcodes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from work_rag_orchestrator import content_registry as cr

REQUIRED_PROMPT_KEYS = (
    "classify",
    "hybrid_classify",
    "clarify",
    "rewrite_system",
    "rewrite_user_with_history",
    "rewrite_user_no_history",
    "answer_system",
    "answer_system_new_chat_suffix",
    "answer_system_continue_suffix",
    "entity_priority_footer",
    "followup_system",
    "followup_user",
    "specify_judge",
    "suggest_subspace",
    "compact_summary_system",
    "compact_incremental_user",
    "greeting_system_hello",
    "greeting_system_farewell",
    "greeting_system_thanks",
    "clarify_fallback",
    "hybrid_judge_escalation",
    "faq_verify_system",
    "faq_verify_user",
    "specification_ack",
)

# Modules that must call require_prompt and must not embed long Persian/English LLM prompts.
_NODE_MODULES = (
    "nodes/classify.py",
    "nodes/hybrid_classify.py",
    "nodes/generate_clarification.py",
    "nodes/build_context.py",
    "nodes/format_response.py",
    "nodes/infer_specification.py",
    "rewrite.py",
    "context_compact.py",
    "clients/hybrid_judge.py",
)

_BANNED_LITERAL_SNIPPETS = (
    "Iranian Credit Scoring Company AI Agent",
    "تو خلاصه‌ساز گفتگو هستی",
    "You are a senior safety judge for a Persian financial",
    "دسته‌بندی کن — فقط یک کلمه",
    "شما یک دسته‌بند پیام کاربر هستید",
)


@pytest.fixture(autouse=True)
def _force_vendored(monkeypatch):
    """Force fallback-yaml path so tests do not depend on a live KB."""
    monkeypatch.setattr(cr, "_CACHE", None)
    monkeypatch.setattr(cr, "_CACHE_AT", 0.0)

    def _boom(*_a, **_k):
        raise RuntimeError("kb down (test)")

    monkeypatch.setattr(cr.httpx, "Client", _boom)
    yield
    monkeypatch.setattr(cr, "_CACHE", None)
    monkeypatch.setattr(cr, "_CACHE_AT", 0.0)


def test_vendored_yaml_loads_all_required_prompts():
    payload = cr.refresh_registry(force=True)
    assert payload["source"] == "fallback-yaml"
    prompts = (payload.get("prompts") or {}).get("prompts") or {}
    for key in REQUIRED_PROMPT_KEYS:
        assert key in prompts, key
        assert str(prompts[key]).strip()


def test_require_prompt_formats_placeholders():
    text = cr.require_prompt("classify", text="سلام")
    assert "سلام" in text
    assert "{text}" not in text
    clarify = cr.require_prompt("clarify_fallback", options="چک، تسهیلات")
    assert "چک، تسهیلات" in clarify


def test_node_modules_have_no_banned_prompt_literals():
    root = Path(cr.__file__).resolve().parent
    for rel in _NODE_MODULES:
        path = root / rel
        src = path.read_text(encoding="utf-8")
        for snippet in _BANNED_LITERAL_SNIPPETS:
            assert snippet not in src, f"{rel} still hardcodes {snippet!r}"
        tree = ast.parse(src)
        # Soft check: module should reference require_prompt somewhere
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "require_prompt" in names, f"{rel} does not call require_prompt"
