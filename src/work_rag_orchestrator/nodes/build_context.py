"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing
from ..rewrite import last_exchanges_text, HISTORY_MAX_CHARS
from ..context_compact import compact_history
from ..config import get_settings

log = logging.getLogger(__name__)

# Max context size — 20-chunk generation upgrade: wider recall window for
# hybrid (RRF + CE rerank) retrieval. 9000 chars silently truncated the tail
# of 14-source contexts (10430+ chars) which are routine now, dropping
# on-topic answers and causing spurious "پاسخی یافت نشد" refusals
# (~14000 chars ≈ 4-5k tokens fits the Gemma 8k window with system+prompt).
MAX_CONTEXT_CHARS = 14000
MAX_CHUNKS = 20


_NO_EVIDENCE_MARKER = "پاسخی برای این سوال یافت نشد"


def _strip_notfound_exchanges(messages):
    """Drop assistant no-evidence turns (and their unanswered user question).

    A previous "not found" answer is stale: retrieval quality improves over
    time and the same question can be answered later. Keeping the refusal in
    the [Conversation history] block biases the model to repeat it.
    """
    out = []
    pending_user = None
    for msg in messages or []:
        role = msg.get("role")
        if role == "user":
            pending_user = msg
            out.append(msg)
        elif role == "assistant":
            if _NO_EVIDENCE_MARKER in " ".join((msg.get("content") or "").split()):
                if pending_user is not None and out and out[-1] is pending_user:
                    out.pop()
                pending_user = None
            else:
                out.append(msg)
                pending_user = None
        else:
            out.append(msg)
    return out


async def build_context(state: RAGState) -> RAGState:
    """
    Create a bounded system/context message with numbered source IDs.

    Uses retrieved_chunks to build context. Limits total size and chunk count.
    Stores prompt_messages in state for the generation step.
    """
    t0 = time.monotonic()
    request_id = state["request_id"]
    query = state["query"]
    chunks = state["retrieved_chunks"]
    messages = state.get("messages", [])

    # New conversation = no assistant turn yet (stateless API: single user
    # message also counts as new). New chats get greeting + company intro.
    is_new_chat = not any(m.get("role") == "assistant" for m in messages)
    
    # Greeting-only: brief reply, no KB context, no history, no citations.
    greeting_kind = state.get("greeting_only")
    if greeting_kind:
        from ..content_registry import require_prompt

        if greeting_kind == "farewell":
            greeting_system = require_prompt("greeting_system_farewell")
        elif greeting_kind == "thanks":
            greeting_system = require_prompt("greeting_system_thanks")
        else:
            greeting_system = require_prompt("greeting_system_hello")
        state["prompt_messages"] = [
            {"role": "system", "content": greeting_system},
            {"role": "user", "content": query},
        ]
        log.info("Built greeting-only context (%s) for request %s", greeting_kind, request_id)
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        record_timing(state, "build_context", elapsed)
        return state

    # T6 fix: do NOT filter by rerank_score threshold here.  If the retrieval
    # returned chunks (even with rerank 0.0), they are evidence when ctx_rel is
    # high or when they contain subspace/query keywords.  Preserve all
    # retrieved chunks up to MAX_CHUNKS and re-order so keyword-matching chunks
    # survive truncation within the 9000-char budget.
    # Prioritize chunks whose content contains query or subspace keywords
    # (lexical evidence) so they are not dropped when truncating.
    try:
        from .format_response import _has_lexical_evidence as _lex_check

        q_for_sort = (state.get("query") or query or "").strip()
        applied = state.get("applied_subspace")
        # stable partition: lexical matches first, then rest, preserving original RRF/CE order
        lex_first: list[dict] = []
        lex_rest: list[dict] = []
        for c in (chunks or []):
            try:
                if _lex_check(q_for_sort, [c], applied):
                    lex_first.append(c)
                else:
                    lex_rest.append(c)
            except Exception:
                lex_rest.append(c)
        # If any lexical matches exist, put them first; otherwise keep original order
        if lex_first:
            chunks = (lex_first + lex_rest)[:MAX_CHUNKS]
        else:
            chunks = (chunks or [])[:MAX_CHUNKS]
    except Exception:
        chunks = (chunks or [])[:MAX_CHUNKS]

    def _section_label(chunk: dict, fallback_idx: int) -> str:
        src = str(chunk.get("source") or "").lower()
        rank_rrf = chunk.get("rank_rrf")
        rank_ce = chunk.get("rank_ce")
        if src == "rrf":
            n = rank_rrf if isinstance(rank_rrf, int) else fallback_idx
            return f"[RRF-{n}]"
        if src == "ce":
            n = rank_ce if isinstance(rank_ce, int) else fallback_idx
            return f"[CE-{n}]"
        if src == "both":
            if isinstance(rank_rrf, int):
                n = rank_rrf
            elif isinstance(rank_ce, int):
                n = rank_ce
            else:
                n = fallback_idx
            return f"[BOTH-{n}]"
        return f"[{fallback_idx}]"

    # Build context with source-set-labelled sections
    context_parts = ["[Context from Knowledge Base]"]

    for i, chunk in enumerate(chunks, 1):
        source_id = _section_label(chunk, i)
        title = chunk.get("title", "Unknown")
        heading = chunk.get("heading", "")
        content = chunk.get("content", "")
        
        context_entry = f"{source_id} Title: {title}"
        if heading:
            context_entry += f"\nHeading: {heading}"
        context_entry += f"\nContent: {content}"
        
        context_parts.append(context_entry)
    
    context_text = "\n\n".join(context_parts)

    # Opt-in diagnostic (ORCH_DEBUG=1): verify the generation context actually
    # carried source text (empty/short context would surface as LLM "یافت نشد").
    try:
        import os
        if os.environ.get("ORCH_DEBUG") == "1":
            log.info("Built context len=%d head=%r tail=%r", len(context_text), context_text[:160], context_text[-160:])
    except Exception:
        pass

    # Stateless memory: last 2 exchanges (prior turns only, not the current
    # question) as a plain-text block, capped at 1000 chars.
    # Failed exchanges ("not found in KB") are stripped: they carry no
    # information and anchor the model into repeating the refusal even when
    # the current retrieval DID find the answer in another subspace.
    #
    # Auto-compacting: when the estimated prompt tokens exceed the configured
    # threshold (8192-token upstream window), older turns are replaced by a
    # cached rolling LLM summary; the newest turns stay verbatim. Short
    # sessions never trigger the summarizer (zero added latency).
    t_compact = time.monotonic()
    compacted_messages, compact_info = await compact_history(messages)
    compact_ms = round((time.monotonic() - t_compact) * 1000, 1)

    history_parts: list[str] = []
    if compact_info.get("compacted"):
        summary_text = ""
        for m in compacted_messages:
            if m.get("role") == "system" and (m.get("content") or "").startswith("[خلاصهٔ گفتگوی پیشین]"):
                summary_text = m["content"].split("\n", 1)[-1]
                break
        if summary_text:
            history_parts.append(f"[خلاصهٔ گفتگوی پیشین] {summary_text}")
    history_excerpt = last_exchanges_text(
        _strip_notfound_exchanges(compacted_messages), include_current=False, max_chars=1000
    )
    if history_excerpt:
        history_parts.append(history_excerpt)
    history_block = f"[Conversation history]\n" + "\n".join(history_parts) if history_parts else ""
    if compact_info.get("compacted"):
        record_timing(state, "compact", compact_ms)
        state["compaction"] = {
            k: compact_info.get(k)
            for k in ("compacted", "old_turns", "cached", "summary_chars", "estimate_tokens")
        }
        log.info(
            "Compaction for %s: old_turns=%d cached=%s summary_chars=%d est_tokens=%d",
            request_id, compact_info.get("old_turns", 0), compact_info.get("cached"),
            compact_info.get("summary_chars", 0), compact_info.get("estimate_tokens", 0),
        )
    question_part = f"Question: {query}"

    # Keep everything within the prompt budget — truncate KB context first,
    # never the question. The budget leaves headroom for the system prompt,
    # the (possibly injected) entity-data section and the model's answer.
    settings = get_settings()
    _entity_reserve = 9500 if state.get("supplemental") else 0
    _prompt_budget = max(settings.prompt_char_budget - _entity_reserve, 0)
    _ctx_cap = min(MAX_CONTEXT_CHARS, _prompt_budget)
    if history_block:
        budget = _ctx_cap - len(history_block) - len(question_part) - 4  # separators
        if budget < 0:  # pathological: drop history rather than the question
            history_block = ""
            budget = _ctx_cap - len(question_part) - 2
        if len(context_text) > budget:
            context_text = context_text[: max(budget, 0)] + "\n...[truncated]"
    elif len(context_text) > _ctx_cap:
        context_text = context_text[:_ctx_cap] + "\n...[truncated]"
    
    # System message from content registry (KB prompts pack) — never hardcoded here.
    from ..content_registry import require_prompt, registry_source

    system_message = require_prompt("answer_system").rstrip()
    if is_new_chat:
        system_message += " " + require_prompt("answer_system_new_chat_suffix").strip()
    else:
        system_message += " " + require_prompt("answer_system_continue_suffix").strip()
    log.debug("answer_system loaded from content registry source=%s", registry_source())
    
    # Build prompt messages for generation
    if history_block:
        user_content = f"{context_text}\n\n{history_block}\n\n{question_part}"
    else:
        user_content = f"{context_text}\n\n{question_part}"

    # --- Inject entity-list tool-call results into the system message ---
    # When the retrieve node fired the supplement tool (reason codes,
    # shareholders, loan titles, ...), render each returned list as an
    # [Entity Data] block so Gemma (which may not render role="tool" input)
    # can consume the matching dictionary rows.
    if state.get("supplemental"):
        try:
            import json as _json
            import re as _re

            def _compact_entity_content(content: dict) -> dict:
                """Keep structured catalog/score fields; drop bulky dumps."""
                if not isinstance(content, dict):
                    return {"value": content}
                out: dict = {}
                meta = content.get("_meta")
                if isinstance(meta, dict):
                    out["_meta"] = {
                        k: meta.get(k)
                        for k in ("kind", "model", "count", "fields", "score_scale", "subspace_ids")
                        if meta.get(k) is not None
                    }
                for key in ("groups", "color_groups", "records", "bands"):
                    if key in content:
                        out[key] = content[key]
                # Fallback for legacy sheet-only payloads
                if not any(k in out for k in ("records", "bands", "groups", "color_groups")):
                    for k, v in content.items():
                        if k.startswith("_"):
                            continue
                        out[k] = v
                return out

            def _entity_relevance(ent: dict, q: str) -> float:
                name = str(ent.get("list_name") or "")
                desc = str(ent.get("description") or "")
                content = ent.get("content") or {}
                meta = content.get("_meta") if isinstance(content, dict) else {}
                kind = str((meta or {}).get("kind") or "")
                blob = f"{name} {desc} {kind}".lower()
                score = float(ent.get("similarity") or 0.0)
                tokens = [t for t in _re.split(r"[\s،,.؟?!:/()\-]+", (q or "").lower()) if len(t) >= 2]
                score += sum(1.5 for t in tokens if t in blob)
                # Prefer catalogs/scores over incidental lists for list/lookup questions
                if kind in {
                    "shareholders",
                    "loan_catalog",
                    "data_providers",
                    "report_consumers",
                    "score_table",
                }:
                    score += 2.0
                if kind == "loan_catalog" and any(k in (q or "") for k in ("وام", "عناوین", "تسهیلات")):
                    score += 15.0
                if kind == "shareholders" and "سهامدار" in (q or ""):
                    score += 15.0
                if kind == "data_providers" and any(
                    k in (q or "") for k in ("تامین", "دیتا", "نئو", "موبایل")
                ):
                    score += 15.0
                if kind == "report_consumers" and any(
                    k in (q or "") for k in ("استفاده", "لندتک", "ریسلر")
                ):
                    score += 15.0
                if kind == "score_table" and any(
                    k in (q or "") for k in ("وام", "عناوین", "سهامدار", "تامین", "لندتک", "ریسلر")
                ) and not any(k in (q or "") for k in ("امتیاز", "رتبه", "رنگ", "گرید")):
                    score -= 20.0
                if any(k in (q or "") for k in ("سهامدار", "وام", "تامین", "لندتک", "ریسلر", "امتیاز", "رتبه", "رنگ")):
                    if any(k in name for k in ("سهامدار", "وام", "تامین", "استفاده", "امتیاز")):
                        score += 3.0
                return score

            ranked = sorted(
                list(state["supplemental"] or []),
                key=lambda e: _entity_relevance(e if isinstance(e, dict) else {}, query),
                reverse=True,
            )
            ent_blocks = []
            # Hard budget shared across entity lists (Persian chars ≈ tokens*2-3).
            _total_budget = 9000
            _used = 0
            for ent in ranked:
                if not isinstance(ent, dict):
                    continue
                content = _compact_entity_content(ent.get("content") or {})
                data_json = _json.dumps(content, ensure_ascii=False)
                desc = str(ent.get("description") or "")
                if len(desc) > 220:
                    desc = desc[:220] + "…"
                header = f"List: {ent.get('list_name')}\nDescription: {desc}\nData: "
                room = _total_budget - _used - len(header)
                if room < 300:
                    break
                if len(data_json) > room:
                    data_json = data_json[:room] + "…(truncated)"
                block = header + data_json
                _used += len(block)
                ent_blocks.append(block)
            if not ent_blocks:
                raise RuntimeError("no room for entity data")
            entity_section = (
                "\n\n[Entity Data from supplement tool]\n"
                "Use this entity data as authoritative for list/lookup answers "
                "(shareholders, loans, providers, report consumers, score tables). "
                "Do not say «یافت نشد» when the answer is present in Data.\n\n"
                + "\n\n---\n\n".join(ent_blocks)
                + "\n\n"
                + require_prompt("entity_priority_footer").strip()
            )
            system_message += entity_section

            # Also prepend to user content for extra prominence when KB chunks may not be relevant
            if ent_blocks:
                user_content = entity_section + "\n\n" + user_content
        except Exception:
            pass

    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
    
    state["prompt_messages"] = prompt_messages
    log.info("Built context with %d sources for request %s", len(chunks), request_id)

    try:
        from ..tracing import trace_span
        trace_span(
            request_id,
            "build_context",
            span_input={"query": query, "chunks": len(chunks), "is_new_chat": is_new_chat, "history_chars": len(history_block)},
            span_output="\n---\n".join(m.get("content", "") for m in prompt_messages),
            metadata={"max_chunks": MAX_CHUNKS, "max_chars": MAX_CONTEXT_CHARS},
            clip=6000,
        )
    except Exception:
        pass

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
        state["stage_timing_ms"] = {}
    state["stage_timing_ms"]["build_context"] = elapsed

    return state