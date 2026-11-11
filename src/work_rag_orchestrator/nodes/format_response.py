"""Format response node - returns OpenAI-compatible output with citation metadata."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from ..state import RAGState, record_timing
from ..schemas import Citation

log = logging.getLogger(__name__)

FOLLOWUP_MAX_TOKENS = 120


def _answer_from_supplemental(state: RAGState) -> str:
    """Deterministic fallback when Entity Data is present but the LLM abstained.

    Catalogs (loans, shareholders, providers, consumers) and score tables are
    authoritative once retrieved; do not surface «یافت نشد» over them.
    """
    ents = state.get("supplemental") or state.get("injected_entities") or []
    if not ents:
        return ""
    q = (state.get("query") or state.get("rewritten_query") or "").strip()
    q_low = q.lower()
    tokens = [t for t in re.split(r"[\s،,.؟?!:/()\-]+", q_low) if len(t) >= 2]

    def _kind(ent: dict) -> str:
        content = ent.get("content") or {}
        meta = content.get("_meta") if isinstance(content, dict) else {}
        return str((meta or {}).get("kind") or "")

    catalog_q = any(
        k in q
        for k in (
            "وام",
            "عناوین وام",
            "سهامدار",
            "تامین",
            "لندتک",
            "ریسلر",
            "استفاده‌کننده",
            "استفاده کننده",
            "نئوبانک",
            "موبایل‌بانک",
            "موبایل بانک",
        )
    )
    score_q = any(
        k in q for k in ("امتیاز", "رتبه", "رنگ", "گرید", "جدول امتیاز", "نکول", "A1", "B1", "C1")
    ) and not catalog_q

    def _score(ent: dict) -> float:
        name = str(ent.get("list_name") or "")
        kind = _kind(ent)
        s = float(ent.get("similarity") or 0.0)
        blob = f"{name} {kind}".lower()
        s += sum(1.5 for t in tokens if t in blob)
        if kind == "loan_catalog" and any(k in q for k in ("وام", "تسهیلات", "عناوین")):
            s += 12.0
        if kind == "shareholders" and "سهامدار" in q:
            s += 12.0
        if kind == "data_providers" and any(
            k in q for k in ("تامین", "دیتا", "داده", "نئو", "موبایل")
        ):
            s += 12.0
        if kind == "report_consumers" and any(
            k in q for k in ("استفاده", "لندتک", "ریسلر", "گزارش")
        ):
            s += 12.0
        if kind == "score_table":
            if catalog_q and not score_q:
                s -= 25.0
            elif score_q:
                s += 12.0
        return s

    ranked = sorted(
        [e for e in ents if isinstance(e, dict)],
        key=_score,
        reverse=True,
    )
    if not ranked:
        return ""
    # Prefer a catalog entity for catalog questions even if ranked second
    ent = ranked[0]
    if catalog_q and _kind(ent) == "score_table":
        for cand in ranked[1:]:
            if _kind(cand) in (
                "loan_catalog",
                "shareholders",
                "data_providers",
                "report_consumers",
            ):
                ent = cand
                break
    content = ent.get("content") or {}
    if not isinstance(content, dict):
        return ""
    kind = _kind(ent)
    name = str(ent.get("list_name") or "")
    groups = content.get("groups") or content.get("color_groups") or []
    records = content.get("records") or content.get("bands") or []

    # Token-matched rows for lookup questions (rank by distinctive hits)
    _STOP = {
        "در", "از", "که", "یا", "نه", "هست", "آیا", "چه", "نوع", "فهرست", "لیست",
        "وجود", "دارد", "عناوین", "جدول", "کدام", "نام", "ببر", "گزارش", "شرکت",
        "اعتبارسنجی", "ایران", "های", "هایی", "این", "آن", "با", "برای", "هم",
    }
    sig_tokens = [t for t in tokens if t not in _STOP and len(t) >= 3]

    def _row_blob(row) -> str:
        if isinstance(row, dict):
            return " ".join(str(v) for v in row.values() if v is not None).lower()
        return str(row).lower()

    scored_rows: list[tuple[float, object]] = []
    for row in records:
        blob = _row_blob(row)
        if not sig_tokens:
            continue
        hits = sum(1 for t in sig_tokens if t in blob)
        # Strong boost when a proper-name token (e.g. تجارت) hits
        name_hits = sum(1 for t in sig_tokens if len(t) >= 4 and t in blob)
        if hits >= 2 or name_hits >= 1:
            scored_rows.append((hits + 2.0 * name_hits, row))
    scored_rows.sort(key=lambda x: x[0], reverse=True)
    matched = [r for _, r in scored_rows]

    listing = any(
        k in q for k in ("چه نوع", "چه رنگ", "نام ببر", "کدام‌ها", "انواع")
    ) or (
        any(k in q for k in ("فهرست", "لیست", "جدول"))
        and not any(k in q for k in ("آیا", "هست یا نه", "یا نه"))
    )
    lookup = any(k in q for k in ("آیا", "هست یا نه", "یا نه", "هست؟", "هست؟")) or (
        any(k in q for k in ("هست", "وجود دارد")) and not listing
    )
    if kind == "loan_catalog":
        # Type listing always wins over a loose row match
        if listing and groups and not (lookup and matched and scored_rows and scored_rows[0][0] >= 4):
            parts = [f"{g.get('گروه')} ({g.get('تعداد')})" for g in groups[:12] if isinstance(g, dict)]
            return (
                f"در «{name}» این انواع وام ثبت شده است: "
                + "، ".join(parts)
                + f". مجموع عناوین: {(content.get('_meta') or {}).get('count') or len(records)}."
            )
        if matched and (lookup or not listing):
            row = matched[0]
            return (
                f"بله، «{row.get('نام وام')}» در فهرست عناوین وام‌ها وجود دارد "
                f"و نوع آن «{row.get('نوع وام')}» است."
            )
        if groups:
            parts = [f"{g.get('گروه')} ({g.get('تعداد')})" for g in groups[:12] if isinstance(g, dict)]
            return (
                f"در «{name}» این انواع وام ثبت شده است: "
                + "، ".join(parts)
                + f". مجموع عناوین: {(content.get('_meta') or {}).get('count') or len(records)}."
            )
    if kind == "shareholders":
        if matched and (lookup or not listing):
            row = matched[0]
            return (
                f"بله، «{row.get('نام سهامدار')}» در فهرست سهامداران است "
                f"(نوع: {row.get('نوع سهامدار')}، حوزه: {row.get('حوزه فعالیت')})."
            )
        if groups:
            parts = [f"{g.get('گروه')} ({g.get('تعداد')})" for g in groups if isinstance(g, dict)]
            banks = [
                str(r.get("نام سهامدار"))
                for r in records
                if isinstance(r, dict) and "بانک" in str(r.get("نوع سهامدار") or "")
            ][:12]
            return (
                f"سهامداران «{name}» در گروه‌های {('، '.join(parts))} قرار دارند. "
                + (f"نمونه بانک‌ها: {('، '.join(banks))}." if banks else "")
            )
    if kind == "data_providers":
        neo = [
            str(r.get("نام تامین کننده"))
            for r in records
            if isinstance(r, dict)
            and (
                r.get("نئوبانک")
                or r.get("موبایل بانک")
                or r.get("نئوبانک ۲")
                or r.get("موبایل بانک ۲")
            )
        ][:12]
        if neo and any(k in q for k in ("نئو", "موبایل")):
            return (
                "در فهرست تامین‌کنندگان دیتا، این موارد نئوبانک یا موبایل‌بانک دارند: "
                + "، ".join(neo)
                + "."
            )
        if groups:
            parts = [f"{g.get('گروه')} ({g.get('تعداد')})" for g in groups if isinstance(g, dict)]
            msg = f"تامین‌کنندگان دیتا شامل {('، '.join(parts))} هستند."
            if neo:
                msg += " نمونه‌های دارای نئوبانک/موبایل‌بانک: " + "، ".join(neo[:6]) + "."
            return msg
    if kind == "report_consumers":
        if matched and (lookup or not listing):
            row = matched[0]
            return (
                f"بله، «{row.get('نام شرکت')}» در فهرست استفاده‌کنندگان گزارش است "
                f"(کانال: {row.get('نوع کانال')}، قرارداد: {row.get('نوع قرارداد')})."
            )
        if lookup and not matched:
            return "در فهرست استفاده‌کنندگان گزارش، شرکتی با این نام ذکر نشده است."
        if groups:
            parts = [f"{g.get('گروه')} ({g.get('تعداد')})" for g in groups if isinstance(g, dict)]
            return f"استفاده‌کنندگان گزارش بر اساس کانال: {('، '.join(parts))}."
    if kind == "score_table" and not catalog_q:
        if groups:
            parts = []
            for g in groups:
                if not isinstance(g, dict):
                    continue
                color = g.get("رنگ") or g.get("color")
                grades = g.get("رتبه‌ها") or g.get("grades")
                lo = g.get("حد پایین امتیاز گروه") or g.get("score_low")
                hi = g.get("حد بالای امتیاز گروه") or g.get("score_high")
                parts.append(f"{color}: {grades} ({lo} تا {hi})")
            if parts:
                return f"جدول امتیاز «{name}»: " + "؛ ".join(parts)
        if matched:
            row = matched[0]
            if isinstance(row, dict):
                return (
                    f"رتبه {row.get('grade') or row.get('رتبه اعتباری')}: "
                    f"رنگ {row.get('color') or row.get('رنگ')}، "
                    f"بازه {row.get('score_low') or row.get('حد پایین امتیاز')} تا "
                    f"{row.get('score_high') or row.get('حد بالای امتیاز')}."
                )
    # Negative lookup: entity present but no row matched a specific name probe
    if matched:
        row = matched[0]
        if isinstance(row, dict):
            vals = " | ".join(str(v) for v in row.values() if v)
            return f"بر اساس فهرست «{name}»: {vals}."
    return ""


async def _generate_grounded_followups(state: RAGState, answer: str) -> list[str]:
    """Generate 2-3 grounded follow-up questions from retrieved_chunks titles."""
    try:
        query = (state.get("query") or state.get("rewritten_query") or "").strip()
        chunks = state.get("retrieved_chunks") or []
        if not query or not answer or not chunks:
            return []
        # Skip Task prompts (WebUI ungrounded follow-up generation)
        if "### Task" in query or "Suggest 3-5" in query:
            return []
        titles: list[str] = []
        for c in chunks[:5]:
            t = str(c.get("title") or "").strip()
            h = str(c.get("heading") or "").strip()
            label = f"{t} — {h}" if t and h else (t or h)
            if label:
                titles.append(label)
        if not titles:
            return []
        titles_str = "\n".join(f"- {x}" for x in titles)
        from ..content_registry import require_prompt
        from ..clients.guardrails import GuardrailsClient
        from ..schemas import GuardrailsChatRequest
        from ..config import get_settings
        settings = get_settings()
        prompt = require_prompt(
            "followup_user",
            query=query[:300],
            answer=answer[:500],
            titles=titles_str,
        )
        req = GuardrailsChatRequest(
            model=settings.upstream_llm_model,
            messages=[
                {"role": "system", "content": require_prompt("followup_system")},
                {"role": "user", "content": prompt},
            ],
            max_tokens=FOLLOWUP_MAX_TOKENS,
            temperature=0.3,
        )
        async with GuardrailsClient(timeout=12.0) as client:
            resp = await asyncio.wait_for(
                client.chat_completion(req, (state.get("request_id") or "") + "-followup"),
                timeout=10.0,
            )
        content = ""
        if resp and getattr(resp, "choices", None):
            msg = resp.choices[0].message  # type: ignore
            if isinstance(msg, dict):
                content = str(msg.get("content") or "")
            else:
                # Pydantic ChatCompletionChoice.message is Dict[str,str]
                try:
                    content = str(msg.get("content") or "")  # type: ignore
                except Exception:
                    content = str(msg)
        if not content:
            return []
        # Extract JSON array substring
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            return []
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in arr:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            if len(s) > 150:
                s = s[:150].strip()
            if not s.endswith("؟") and not s.endswith("?"):
                s = s + "؟"
            low = s.lower().strip()
            if low in seen or low == query.lower().strip():
                continue
            seen.add(low)
            cleaned.append(s)
            if len(cleaned) >= 3:
                break
        # Require 2-3, but return whatever we have (0-3)
        return cleaned[:3]
    except asyncio.TimeoutError:
        log.warning("grounded follow-ups timeout for %s", state.get("request_id"))
        return []
    except Exception as e:
        log.warning("grounded follow-ups failed for %s: %s", state.get("request_id"), e)
        return []


_CLEAN_PATTERNS = [
    (re.compile(r"<unused\d+>"), ""),
    (re.compile(r"<\|?tool_call\|?>"), ""),
    (re.compile(r"<\|?tool_response\|?>"), ""),
    (re.compile(r"tool_response\|>"), ""),
    (re.compile(r"tool_call\|>"), ""),
    (re.compile(r"\[multimodal\]"), ""),
    (re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL), ""),
    (re.compile(r"<\|think\|>"), ""),
    (re.compile(r"<\|turn>.*?<turn\|>", re.DOTALL), ""),
    (re.compile(r"<bos>"), ""),
    (re.compile(r"<eos>"), ""),
    (re.compile(r"<\|?tool\|?>"), ""),
    (re.compile(r"<\|\s*\"\s*\|>"), ""),
    (re.compile(r"<\|\s*'\s*\|>"), ""),
    (re.compile(r"<\|[^>]*\|>"), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n[ \t]*\n[ \t]*\n+"), "\n\n"),
    (re.compile(r"(?<!\n)\n(?!\n)"), " "),
]


def _clean_answer(text: str) -> str:
    if not text:
        return text
    for pat, repl in _CLEAN_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()


def _to_plain_text(text: str) -> str:
    """Force plain Persian text: drop markdown and [n] citation markers.

    The model is instructed not to emit them, but this is defensive:
    stray '*' bullets, '**bold**', '#' headers, backticks and [n] markers
    are removed so the user always sees clean plain text. Citation metadata
    is preserved separately in `rag.citations`.
    """
    import re
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold** -> bold
    text = re.sub(r"__([^_]+?)__", r"\1", text)    # __bold__ -> bold
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)  # # headers
    text = re.sub(r"(?m)^\s*[*\-•]\s+", "", text)  # bullet line markers
    text = text.replace("`", "")
    text = re.sub(r"\[\d+\]", "", text)            # [1] citation markers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MAX_CITATIONS = 5

MAX_GROUND_TRUTH_LINES = 20


def _ground_truth_header() -> str:
    from ..content_registry import canned_message

    return canned_message("ground_truth_header")


def _no_evidence_msg() -> str:
    from ..content_registry import refusal_message

    return refusal_message("no_evidence")


def _abstention_phrase() -> str:
    from ..content_registry import abstention_phrase

    return abstention_phrase()


# F: faith/hallucination guard thresholds for weak-evidence answers. An
# answer produced on a weak-evidence path (T1 whole-KB fallback or T2
# broadened retry) is replaced with NO_EVIDENCE when the kept/cited
# chunks also score below FAITH_GUARD_MIN_SCORE and the answer's lexical
# coverage over the kept chunk text is below FAITH_GUARD_MIN_COVERAGE —
# both low means the generated text is not grounded in the evidence
# (hallucination rate was 0.244 on that path).
FAITH_GUARD_MIN_SCORE = 0.5
FAITH_GUARD_MIN_COVERAGE = 0.20


def _source_label(chunk: dict, fallback_idx: int) -> str:
    """Source-set label with the relevant rank, e.g. [RRF-3], [CE-2], [BOTH-1]."""
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


def _chunk_score(chunk: dict) -> float | None:
    for key in ("hybrid_score", "rerank_score", "score"):
        v = chunk.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _ground_truth_block(chunks: list) -> str:
    """Plain-text block listing retrieved sources (no markdown)."""
    lines = []
    for i, c in enumerate(chunks[:MAX_GROUND_TRUTH_LINES], 1):
        title = str(c.get("title") or "").strip()
        heading = str(c.get("heading") or "").strip()
        if not title and not heading:
            continue
        label = _source_label(c, i)
        score = _chunk_score(c)
        if score is not None:
            from ..content_registry import canned_message
            score_txt = canned_message("score_suffix", score=round(score, 3))
        else:
            score_txt = ""
        if title and heading:
            lines.append(f"{label} {title} — {heading}{score_txt}")
        elif title:
            lines.append(f"{label} {title}{score_txt}")
        else:
            lines.append(f"{label} {heading}{score_txt}")
    if not lines:
        return ""
    return _ground_truth_header() + "\n".join(lines)


def _citation_priority(chunk: dict) -> int:
    src = str(chunk.get("source") or "").lower()
    return {"both": 0, "ce": 1, "rrf": 2}.get(src, 3)


# F2: Patterns that indicate retrieved content says something is NOT available
# Be specific: only match explicit "not available for X" or "not defined for X" patterns
# Avoid generic "ندارد", "نیست" which appear in normal content
_NOT_AVAILABLE_PATTERNS = [
    "تعریف نشده", "تعریف نشده است", "تعریف نشده‌اند",
    "ارائه نمی‌شود", "ارائه نشده", "ارائه نشده است",
    "ارائه نمی‌گردد", "ارائه نگردیده",
    "پشتیبانی نمی‌شود", "پشتیبانی نشده",
    "فعال نیست", "فعال نشده",
    "در آینده", "به طور کلی.*نمی‌شود", "به طور کلی.*ندارد",
    # Explicit "not for legal entities" / "not for companies"
    "برای اشخاص حقوقی.*تعریف نشده", "برای شرکت‌ها.*تعریف نشده",
    "برای اشخاص حقوقی.*موجود نیست", "برای شرکت‌ها.*موجود نیست",
    "فقط برای اشخاص حقیقی", "تنها برای اشخاص حقیقی",
    "مختص اشخاص حقیقی", "اختصاصاً برای اشخاص حقیقی",
]

# Patterns that indicate generated answer provides advice/instructions
# as if the thing DOES exist
_ADVICE_PATTERNS = [
    "برای بهبود", "برای ارتقا", "برای افزایش",
    "لازم است", "باید", "وصیه می‌شود", "توصیه می‌شود",
    "اقدام", "مدیریت", "پرهیز", "کاهش", "جلوگیری",
    "ارتقا", "بهبود", "تسویه", "رفع",
    "استراتژی", "راهکار", "روش", "نحوه",
]


async def _check_contradiction(answer: str, chunks: list, request_id: str, applied_subspace: str | None) -> bool:
    """
    Check if answer contradicts retrieved chunks that say something is NOT available.
    
    Returns True if contradiction detected (should refuse).
    """
    # Collect all chunk content
    chunk_text = " ".join(str(c.get("content") or "") for c in chunks[:10])
    if not chunk_text:
        return False
    
    # Check if any chunk says "not available" FOR THE RELEVANT ENTITY TYPE
    chunk_lower = chunk_text.lower()
    
    # Only check for unavailability patterns if the subspace matches
    # For cheque subspace: check if chunks say "not available for legal entities"
    # For facilities_legal: check if chunks say "not available for companies" for cheque
    has_not_available = False
    if applied_subspace == "cheque":
        # For individual cheque: contradiction if chunks say "not for legal entities" 
        # AND answer gives advice as if it's available for legal entities
        # But we don't know if answer is for legal or individual - skip for cheque subspace
        # since cheque IS available for individuals
        has_not_available = False
    elif applied_subspace == "facilities_legal":
        # For company facilities: check if chunks say cheque not available for companies
        has_not_available = any(pat.lower() in chunk_lower for pat in _NOT_AVAILABLE_PATTERNS 
                                if "حقوقی" in pat or "شرکت" in pat or "قانونی" in pat)
    else:
        # For other subspaces, use general patterns but be conservative
        has_not_available = any(pat.lower() in chunk_lower for pat in _NOT_AVAILABLE_PATTERNS 
                                if "تعریف نشده" in pat or "ارائه نمی‌شود" in pat or "پشتیبانی نمی‌شود" in pat)
    
    if not has_not_available:
        return False
    
    # Check if answer gives advice as if it IS available
    answer_lower = answer.lower()
    has_advice = any(pat.lower() in answer_lower for pat in _ADVICE_PATTERNS)
    
    if not has_advice:
        return False
    
    # Additional check: answer should not be a direct quote from chunks
    # (if answer is just repeating the "not available" info, that's fine)
    not_avail_in_answer = any(pat.lower() in answer_lower for pat in _NOT_AVAILABLE_PATTERNS)
    if not_avail_in_answer:
        # Answer acknowledges unavailability - not a contradiction
        return False
    
    log.info(
        "Contradiction guard fired for request %s: chunks say NOT available, but answer gives advice",
        request_id
    )
    return True


# F3: Specific hallucination patterns for subspace mismatches
# e.g., treating a factor as a standalone model
_STANDALONE_MODEL_PATTERNS = [
    "رتبه چک", "امتیاز چک", "رتبه‌بندی چک", "گزارش چک",
    "رتبه اعتباری چک", "امتیاز اعتباری چک",
]

# Patterns that indicate the KB only mentions it as a factor/index
_FACTOR_ONLY_PATTERNS = [
    "شاخص چک", "شاخص‌های چک", "عامل چک", "شاخص \"چک\"",
    "تاثیر چک", "اثر چک", "عامل موثر", "شاخص موثر",
]


async def _check_subspace_mismatch(answer: str, chunks: list, request_id: str, applied_subspace: str | None) -> bool:
    """
    Check for hallucinations where model treats a factor as a standalone model.
    
    E.g., KB says "cheque index is a factor in facilities rating for companies"
    but model answers "how to improve your cheque rating for companies".
    """
    if not applied_subspace or not chunks:
        return False
    
    # Check if applied subspace is facilities_legal and query is about cheque
    if applied_subspace != "facilities_legal":
        return False
    
    chunk_text = " ".join(str(c.get("content") or "") for c in chunks[:10])
    if not chunk_text:
        return False
    
    chunk_lower = chunk_text.lower()
    answer_lower = answer.lower()
    
    # Check if answer talks about standalone cheque rating/model
    has_standalone = any(pat in answer_lower for pat in _STANDALONE_MODEL_PATTERNS)
    if not has_standalone:
        return False
    
    # Check if chunks only mention cheque as a factor/index, not a standalone model
    has_factor_only = any(pat in chunk_lower for pat in _FACTOR_ONLY_PATTERNS)
    has_not_standalone = not any(pat in chunk_lower for pat in ["گزارش چک", "رتبه چک", "امتیاز چک", "رتبه‌بندی چک"])
    
    if has_factor_only and has_not_standalone:
        log.info(
            "Subspace mismatch guard fired for request %s: KB mentions cheque as factor only, but answer treats as standalone model",
            request_id
        )
        return True
    
    return False


def _parse_cited_indices(text: str) -> list[int]:
    """Extract ordered, unique [n] source numbers referenced in the answer."""
    import re
    if not text:
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", text):
        idx = int(m.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _top_evidence_score(chunks: list) -> float:
    """Best rerank/hybrid/score across chunks, 0.0 if none.

    T6 fix: consider MAX across all score keys (not first-found) so a chunk
    with rerank 0.0 but hybrid 0.03 is scored as 0.03 instead of 0.0.
    """
    best = 0.0
    for c in chunks or []:
        for key in ("rerank_score", "hybrid_score", "score"):
            v = c.get(key)
            if isinstance(v, (int, float)):
                if float(v) > best:
                    best = float(v)
        # also consider injected supplemental similarity as evidence
        # (handled separately in retrieve supplement fallback)
    return best


def _has_lexical_evidence(query: str, chunks: list, applied_subspace: str | None) -> bool:
    """T6: lexical proxy for ctx_rel — does any chunk contain query/subspace keywords?

    The cross-encoder rerank gives 0.0 for these turns even though the LLM judge
    scores ctx_rel 1.0.  When the numeric score is low but the text clearly
    overlaps the query, we must not treat the context as 'no evidence' and
    must not abstain.  Checks:
    - subspace keyword (چک, تسهیلات, etc.) in any chunk content
    - any informative query token (len>=2) appears in any chunk
    """
    if not chunks:
        return False
    q = (query or "").strip()
    if not q:
        return False
    # subspace keyword map (same as infer_specification)
    subspace_kw = {
        "cheque": ["چک"],
        "facilities_person": ["تسهیلات", "وام", "قسط", "حقیقی"],
        "facilities_legal": ["حقوقی", "شرکت"],
        "credit_ladder": ["نردبان"],
        "guarantor": ["کفیل", "کفالت"],
        "supplier_legal": ["تامین"],
        "customers_person": ["مشتریان", "پنل"],
        "customers_legal": ["مشتریان", "پنل"],
        "general": ["معرفی", "مدیر", "شرکت", "عمومی", "عام"],
    }
    # collect keywords to test: subspace-specific + query tokens
    keywords: set[str] = set()
    s = (applied_subspace or "").strip().lower()
    # Persian subspace from filter_path also contains "چک" etc.
    if "چک" in (applied_subspace or "") or s == "cheque":
        keywords.update(subspace_kw["cheque"])
    elif "نردبان" in (applied_subspace or "") or s == "credit_ladder":
        keywords.update(subspace_kw["credit_ladder"])
    elif "کفیل" in (applied_subspace or "") or s == "guarantor":
        keywords.update(subspace_kw["guarantor"])
    elif "تامین" in (applied_subspace or "") or s == "supplier_legal":
        keywords.update(subspace_kw["supplier_legal"])
    elif "مشتریان" in (applied_subspace or "") or s in ("customers_person", "customers_legal"):
        keywords.update(subspace_kw["customers_person"])
    elif "حقیقی" in (applied_subspace or "") or s == "facilities_person":
        keywords.update(subspace_kw["facilities_person"])
    elif "حقوقی" in (applied_subspace or "") or s == "facilities_legal":
        keywords.update(subspace_kw["facilities_legal"])
    elif "عمومی" in (applied_subspace or "") or "پایگاه دانش عمومی" in (applied_subspace or "") or s == "general":
        keywords.update(subspace_kw["general"])
    # also add informative query tokens (split on whitespace, strip punctuation)
    import re as _re
    q_norm = _re.sub(r"[؟?!.,،؛:\"'«»()…\-\.]+", " ", q)
    for tok in q_norm.split():
        t = tok.strip()
        if len(t) >= 2 and t not in {"در", "با", "از", "به", "که", "را", "یا", "هم", "این", "آن", "برای", "آیا", "چقدر", "چطور", "چگونه", "چیست", "است"}:
            keywords.add(t)
    if not keywords:
        return False
    # join all chunk contents lower (first 5 chunks are enough, but check all)
    corpus = " ".join(str(c.get("content") or "") for c in chunks[:10])
    if not corpus:
        return False
    for kw in keywords:
        if kw and kw in corpus:
            return True
        # also check without ZWNJ variance
        kw2 = kw.replace("‌", " ").strip()
        if kw2 and kw2 != kw and kw2 in corpus:
            return True
    return False


def _token_coverage(answer: str, chunks: list) -> float:
    """F: lexical coverage of the answer over the kept chunk contents.

    Whitespace tokenization on both sides;
    coverage = |answer_tokens ∩ chunk_tokens| / |answer_tokens|.
    Returns 1.0 for an empty answer (nothing to ground) and 0.0 when the
    kept chunks carry no text.
    """
    a_tokens = set((answer or "").split())
    if not a_tokens:
        return 1.0
    corpus = " ".join(str(c.get("content") or "") for c in (chunks or []))
    c_tokens = set(corpus.split())
    if not c_tokens:
        return 0.0
    return len(a_tokens & c_tokens) / len(a_tokens)

async def format_response(state: RAGState) -> RAGState:
    """
    Return OpenAI-compatible output with citation metadata.
    
    If blocked, returns refusal with content_filter finish_reason.
    If allowed, returns answer with citations from retrieved chunks.
    """
    t0 = time.monotonic()
    request_id = state["request_id"]
    answer_raw = _to_plain_text(_clean_answer(state["answer"]))
    blocked = state.get("blocked", False)
    state_refusal = state.get("refusal_message")
    chunks = state["retrieved_chunks"]
    follow_ups: list[str] = []

    try:
        from ..content_registry import (
            abstention_phrase as registry_abstention,
            greeting_message as registry_greeting,
            refusal_message as registry_refusal,
            subspace_map as registry_subspace_map,
        )
    except Exception:  # pragma: no cover
        registry_greeting = None  # type: ignore
        registry_refusal = None  # type: ignore
        registry_subspace_map = None  # type: ignore
        registry_abstention = None  # type: ignore

    _abstain = _abstention_phrase()

    if blocked:
        from ..content_registry import refusal_message as _rm
        blocked_default = registry_refusal("blocked_default") if registry_refusal else _rm("blocked_default")

        content = _to_plain_text(_clean_answer(state_refusal)) or blocked_default
        finish_reason = "content_filter"
        citations = []
    elif state.get("greeting_only"):
        # Smalltalk: brief reply as-is — no sources retrieved, so no truth
        # block and no citations. Citing nothing is correct here.
        if not answer_raw or not answer_raw.strip() or _abstain in answer_raw:
            kind = state.get("greeting_only") or "greeting"
            from ..content_registry import greeting_message
            answer_raw = greeting_message(kind if kind in ("farewell", "thanks", "greeting") else "greeting")
        content = answer_raw
        finish_reason = "stop"
        citations = []
    elif state.get("classify") == "specification":
        subspace = state.get("applied_subspace") or ""
        from ..content_registry import canned_message, subspace_map as _sub_map
        label_map = registry_subspace_map() if registry_subspace_map else _sub_map()
        label = label_map.get(subspace, subspace) or canned_message("domain_label_fallback")
        if not answer_raw or not answer_raw.strip() or _abstain in answer_raw:
            answer_raw = canned_message("specification_ack", label=label)
        content = answer_raw
        finish_reason = "stop"
        citations = []
    elif state.get("direct_faq_answer") and state.get("faq_verified"):
        if not answer_raw or not answer_raw.strip():
            from ..content_registry import refusal_message as _rm
            answer_raw = registry_refusal("weak_evidence") if registry_refusal else _rm("weak_evidence")

        content = answer_raw
        finish_reason = "stop"
        citations = []
    else:
        # --- H1: post-abstention broadened re-retrieve + single regenerate ---
        # Gemma abstained ("یافت نشد") even though KB may hold on-topic content
        # that the (possibly history-rewritten) query failed to surface (e.g.
        # war-period/date phrasing ranked below the cut). Retry exactly once
        # against the whole KB (filter_path=None) with the ORIGINAL user query;
        # after one attempt the canonical refusal stands.
        if (
            _abstention_phrase() in answer_raw
            and not state.get("broaden_after_abstention")
            and not state.get("broaden_used")
        ):
            try:
                state = await _broaden_retrieve_and_regenerate(state)
                answer_raw = _to_plain_text(_clean_answer(state.get("answer") or ""))
                chunks = state["retrieved_chunks"]
            except Exception as e:
                log.warning(
                    "H1 post-abstention broaden regen failed for %s: %s", request_id, e
                )

        # Prefer Entity Data over abstention / empty / score-table bleed.
        _entity_recovery = ""
        try:
            from ..content_registry import refusal_message as _rm_chk

            _weak_msg = _rm_chk("weak_evidence")
        except Exception:
            _weak_msg = ""
        _q_now = state.get("query") or state.get("rewritten_query") or ""
        _catalog_intent = any(
            k in _q_now
            for k in (
                "وام",
                "عناوین وام",
                "سهامدار",
                "تامین",
                "لندتک",
                "ریسلر",
                "استفاده‌کننده",
                "نئوبانک",
                "موبایل‌بانک",
                "موبایل بانک",
            )
        )
        _score_bleed = any(
            k in (answer_raw or "")
            for k in ("جدول امتیاز", "آبی:", "سبز:", "نارنجی:", "قرمز:")
        ) and _catalog_intent
        # Authoritative dictionary lookups: prefer Entity Data over free-form LLM
        _dict_intent = any(
            k in _q_now
            for k in (
                "عناوین وام",
                "فهرست عناوین",
                "لیست و عناوین",
                "تامین‌کنندگان دیتا",
                "تامین کنندگان دیتا",
                "نئوبانک",
                "موبایل‌بانک",
                "موبایل بانک",
            )
        )
        # Quoted proper-name probe against a catalog (e.g. «آژانس فضایی مریخ»)
        _named_probe = bool(re.search(r"[«\"].+[»\"]", _q_now)) and any(
            k in _q_now for k in ("استفاده‌کننده", "سهامدار", "تامین", "وام")
        )
        if (
            (not answer_raw or not answer_raw.strip())
            or (_abstention_phrase() in (answer_raw or ""))
            or (_weak_msg and _weak_msg in (answer_raw or ""))
            or _score_bleed
            or _dict_intent
            or _named_probe
        ):
            _entity_recovery = _answer_from_supplemental(state)
            if _entity_recovery:
                log.info(
                    "Entity-data recovery for request %s (%d chars; bleed=%s dict=%s)",
                    request_id,
                    len(_entity_recovery),
                    _score_bleed,
                    _dict_intent,
                )
                answer_raw = _entity_recovery
                state["entity_data_recovery"] = True

        # --- G5 hallucination guard: refuse when evidence is too weak ---
        # "رتبه اعتباری G5 چیه" returns 1 noise chunk with rerank 0.008;
        # without this guard the LLM synthesizes "یکی از سطوح...".
        # If top rerank/hybrid/score < threshold and no supplemental entity,
        # treat as no evidence and return the canonical "پاسخی یافت نشد".
        try:
            from ..config import get_settings as _get_settings_ev
            _threshold = float(_get_settings_ev().get_evidence_threshold(state.get("applied_subspace")))
        except Exception:
            _threshold = 0.15
        _top = _top_evidence_score(chunks)
        _has_supplement = bool(state.get("injected_entities") or state.get("supplemental"))
        # T6: if lexical evidence exists (query/subspace keyword in chunk content),
        # don't treat low rerank_score as no-evidence — ctx_rel high despite 0.0 score
        _q_for_lex = state.get("query") or state.get("rewritten_query") or ""
        _has_lex = _has_lexical_evidence(_q_for_lex, chunks, state.get("applied_subspace"))
        if (not chunks or _top < _threshold) and not _has_supplement and not _has_lex:
            log.info(
                "No evidence for request %s: top_score=%.4f < threshold=%.3f -> refusing (G5 guard)",
                request_id, _top, _threshold,
            )
            content = _no_evidence_msg()
            finish_reason = "stop"
            citations = []
            follow_ups = []
            state["follow_ups"] = follow_ups
            state["formatted_response"] = {
                "content": content,
                "finish_reason": finish_reason,
                "citations": [],
                "follow_ups": follow_ups,
            }
            log.info("Formatted response (no-evidence guard) for request %s", request_id)
            try:
                from ..tracing import trace_span as _trace_span
                _trace_span(
                    request_id,
                    "format_response",
                    span_input={"blocked": blocked, "finish_reason": finish_reason, "guard": "no_evidence", "top_score": _top, "threshold": _threshold},
                    span_output={"content": content, "citations": 0},
                    clip=3000,
                )
            except Exception:
                pass
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            record_timing(state, "format_response", elapsed)
            return state

        if not answer_raw or not answer_raw.strip():
            from ..content_registry import refusal_message as _rm

            answer_raw = _rm("weak_evidence")
        content = answer_raw
        finish_reason = "stop"
        # Plain-text answers carry no [n] markers, so cite the top-ranked
        # chunks (max 5) as metadata, preferring "both" > "ce" > "rrf";
        # still honor explicit [n] if present.
        cited = _parse_cited_indices(content)
        if cited:
            chosen = [chunks[i - 1] for i in cited if 1 <= i <= len(chunks)]
        else:
            ranked = sorted(enumerate(chunks), key=lambda t: (_citation_priority(t[1]), t[0]))
            chosen = [c for _, c in ranked[:MAX_CITATIONS]]
        chosen = chosen[:MAX_CITATIONS]
        # If the (possibly regenerated) answer still abstains, drop citations:
        # a refusal should not pretend to be grounded in displayed sources.
        if _abstention_phrase() in content:
            citations = []
            chosen = []
        citations = [
            Citation(
                chunk_id=c.get("chunk_id", ""),
                document_id=c.get("document_id", ""),
                title=c.get("title", ""),
                heading=c.get("heading", ""),
            )
            for c in chosen
            if c.get("chunk_id")
        ]
        # --- F: faith/hallucination guard on weak-evidence answers ---
        # Single post-generation check (no retry loop): when ALL hold —
        # (a) the answer came from a weak-evidence path (T1 whole-KB
        # fallback or T2 broadened retry), (b) the top score of the
        # kept/cited chunks is < 0.5, and (c) lexical coverage between the
        # answer and the kept chunk contents is < 0.20 — the model is
        # fabricating beyond the evidence, so replace the answer body with
        # the canonical refusal and drop citations. Skipped when the
        # answer already abstains. The decision is logged with the
        # coverage value.
        _faith_refused = False
        if (
            (state.get("fallback_used") or state.get("broaden_used"))
            and _abstention_phrase() not in content
            and not state.get("entity_data_recovery")
        ):
            _used_top = _top_evidence_score(chosen)
            _coverage = _token_coverage(content, chosen)
            if _used_top < FAITH_GUARD_MIN_SCORE and _coverage < FAITH_GUARD_MIN_COVERAGE:
                # If Entity Data can answer, prefer recovery over faith refusal
                _rec = _answer_from_supplemental(state)
                if _rec:
                    log.info(
                        "Faith guard deferred for request %s: using entity-data recovery",
                        request_id,
                    )
                    content = _rec
                    state["entity_data_recovery"] = True
                else:
                    log.info(
                        "Faith guard fired for request %s: coverage=%.3f < %.2f and top_used_score=%.3f < %.2f (fallback_used=%s, broaden_used=%s) -> replacing answer with refusal",
                        request_id, _coverage, FAITH_GUARD_MIN_COVERAGE, _used_top, FAITH_GUARD_MIN_SCORE,
                        bool(state.get("fallback_used")), bool(state.get("broaden_used")),
                    )
                    content = _no_evidence_msg()
                    citations = []
                    _faith_refused = True
            else:
                log.info(
                    "Faith guard passed for request %s: coverage=%.3f, top_used_score=%.3f (fallback_used=%s, broaden_used=%s)",
                    request_id, _coverage, _used_top,
                    bool(state.get("fallback_used")), bool(state.get("broaden_used")),
                )

        # --- F2: contradiction guard ---
        # Detect when retrieved chunks explicitly state something is NOT available
        # (e.g., "not defined for legal entities", "not available", "does not exist")
        # but the model generates advice as if it DOES exist.
        # This catches hallucinations where model ignores "not available" signals.
        _contradiction_refused = False
        if not _faith_refused and _abstention_phrase() not in content and chunks:
            try:
                _contradiction_refused = await _check_contradiction(content, chunks, request_id, state.get("applied_subspace"))
                if _contradiction_refused:
                    content = _no_evidence_msg()
                    citations = []
            except Exception as _e:
                log.warning("Contradiction check failed for %s: %s", request_id, _e)

        # --- F3: subspace mismatch guard ---
        # Detect hallucinations where model treats a factor as a standalone model.
        # E.g., KB says "cheque index is a factor in facilities rating for companies"
        # but model answers "how to improve your cheque rating for companies".
        _mismatch_refused = False
        if not _faith_refused and not _contradiction_refused and _abstention_phrase() not in content and chunks:
            try:
                _mismatch_refused = await _check_subspace_mismatch(
                    content, chunks, request_id, state.get("applied_subspace")
                )
                if _mismatch_refused:
                    content = _no_evidence_msg()
                    citations = []
            except Exception as _e:
                log.warning("Subspace mismatch check failed for %s: %s", request_id, _e)
        # Append the ground-truth block AFTER cleaning so _to_plain_text /
        # _clean_answer never strip it; the block itself is plain text.
        # Production responses hide it (response_detail setting); verbose shows.
        # A faith refusal carries no sources — skip the block for refusals.
        from ..config import get_settings as _get_settings
        try:
            _verbose = _get_settings().response_detail.strip().lower() == "verbose"
        except Exception:
            _verbose = False
        if _verbose and not _faith_refused:
            block = _ground_truth_block(chunks)
            if block:
                content = content + block
        # --- Grounded follow-ups: only for successful RAG answers with citations ---
        # Skip for blocked/greeting/spec/faq/no_evidence. Generates 2-3 questions
        # that ARE answerable from the same retrieved chunks (same leaf).
        if finish_reason == "stop" and citations and chunks and content != _no_evidence_msg():
            try:
                _q = state.get("query") or ""
                if "### Task" not in _q and "Suggest 3-5" not in _q:
                    follow_ups = await _generate_grounded_followups(state, content)
                else:
                    follow_ups = []
            except Exception as e:
                log.warning("follow_ups generation error for %s: %s", request_id, e)
                follow_ups = []
        else:
            follow_ups = []
    
    # Store formatted response data in state for API layer (including grounded follow_ups)
    state["follow_ups"] = follow_ups
    state["formatted_response"] = {
        "content": content,
        "finish_reason": finish_reason,
        "citations": [c.model_dump() for c in citations],
        "follow_ups": follow_ups,
    }
    
    log.info("Formatted response for request %s", request_id)

    try:
        from ..tracing import trace_span
        trace_span(
            request_id,
            "format_response",
            span_input={"blocked": blocked, "finish_reason": finish_reason},
            span_output={"content": content, "citations": len(citations)},
            clip=3000,
        )
    except Exception:
        pass

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    record_timing(state, "format_response", elapsed)

    return state


async def _broaden_retrieve_and_regenerate(state: RAGState) -> RAGState:
    """
    H1: after a generation-time "یافت نشد", retry retrieval once against the
    whole KB (filter_path=None) using the ORIGINAL user query, then regenerate.
    Used when the history-rewritten query drifted away from matching phrasing
    (e.g. dates/war terms) so on-topic chunks ranked below the cut. Single
    attempt only — guarded by broaden_after_abstention / broaden_used flags.
    """
    request_id = state["request_id"]
    query = (state.get("query") or state.get("rewritten_query") or "").strip()
    log.info(
        "H1 broaden-regenerate for request %s: query=%r", request_id, query[:80]
    )
    from ..clients.knowledgebase import KnowledgebaseClient as _KBClient

    async with _KBClient() as client:
        result = await client.retrieve(query, 20, request_id, filter_path=None, keyword_boost=1.5)

    fb_chunks: list[dict] = []
    if isinstance(result, dict):
        for item in (result.get("rrf") or []) + (result.get("ce") or []):
            fb_chunks.append(
                {
                    "chunk_id": getattr(item, "chunk_id", None)
                    or item.get("chunk_id", ""),
                    "document_id": getattr(item, "document_id", None)
                    or item.get("document_id", ""),
                    "title": getattr(item, "title", None) or item.get("title", ""),
                    "heading": getattr(item, "heading", None) or item.get("heading", ""),
                    "content": getattr(item, "content", None)
                    or item.get("content", ""),
                    "score": getattr(item, "score", None)
                    or item.get("score", 0.0),
                    "source": getattr(item, "source", None)
                    or item.get("source", "broaden"),
                    "rank_rrf": getattr(item, "rank_rrf", None),
                    "rank_ce": getattr(item, "rank_ce", None),
                    "hybrid_score": getattr(item, "hybrid_score", None),
                    "rerank_score": getattr(item, "rerank_score", None),
                }
            )

    # De-duplicate by chunk_id, preserve merged (RRF then CE) order.
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in fb_chunks:
        cid = c.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            deduped.append(c)
    fb_chunks = deduped[:20]

    state["broaden_after_abstention"] = True
    state["broaden_used"] = True
    state["fallback_used"] = True
    base_reason = state.get("fallback_reason") or "generation abstained"
    if fb_chunks:
        state["retrieved_chunks"] = fb_chunks
        state["fallback_reason"] = (
            f"{base_reason} | H1 broaden regen: len={len(fb_chunks)}"
        )
        log.info(
            "H1 broaden-regenerate adopted %d chunks for request %s",
            len(fb_chunks),
            request_id,
        )
    else:
        state["fallback_reason"] = f"{base_reason} | H1 broaden regen: empty"
        log.info("H1 broaden-regenerate empty for request %s", request_id)

    # Rebuild context from the broadened chunks and regenerate once.
    from .build_context import build_context as _bc
    from .guarded_generate import guarded_generate as _gg

    state = await _bc(state)
    state = await _gg(state)
    return state