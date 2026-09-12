"""Format response node - returns OpenAI-compatible output with citation metadata."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..schemas import Citation

log = logging.getLogger(__name__)


def _clean_answer(text: str) -> str:
    import re
    if not text:
        return text
    text = re.sub(r"<unused\d+>", "", text)
    text = re.sub(r"<\|?tool_call\|?>", "", text)
    text = re.sub(r"<\|?tool_response\|?>", "", text)
    text = re.sub(r"tool_response\|>", "", text)
    text = re.sub(r"tool_call\|>", "", text)
    text = re.sub(r"\[multimodal\]", "", text)
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|think\|>", "", text)
    text = re.sub(r"<\|turn>.*?<turn\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<bos>", "", text)
    text = re.sub(r"<eos>", "", text)
    text = re.sub(r"<\|?tool\|?>", "", text)
    text = re.sub(r"<\|\s*\"\s*\|>", "", text)
    text = re.sub(r"<\|\s*'\s*\|>", "", text)
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


MAX_CITATIONS = 2


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

async def format_response(state: RAGState) -> RAGState:
    """
    Return OpenAI-compatible output with citation metadata.
    
    If blocked, returns refusal with content_filter finish_reason.
    If allowed, returns answer with citations from retrieved chunks.
    """
    request_id = state["request_id"]
    answer = _to_plain_text(_clean_answer(state["answer"]))
    # Fallback if cleaning left empty (model only emitted control tokens)
    if not answer or not answer.strip():
        answer = "بر اساس منابع بازیابی‌شده، پاسخ مستقیم در متن موجود نیست؛ لطفاً سوال را دقیق‌تر بپرسید."
    blocked = state.get("blocked", False)
    refusal_message = state.get("refusal_message")
    chunks = state["retrieved_chunks"]

    if blocked:
        content = _to_plain_text(_clean_answer(refusal_message)) or "I cannot comply with that request."
        finish_reason = "content_filter"
        citations = []
    else:
        content = answer
        finish_reason = "stop"
        # Plain-text answers carry no [n] markers, so cite the top-ranked
        # chunks (max 2) as metadata; still honor explicit [n] if present.
        cited = _parse_cited_indices(content)
        if cited:
            chosen = [chunks[i - 1] for i in cited if 1 <= i <= len(chunks)]
        else:
            chosen = chunks[:MAX_CITATIONS]
        chosen = chosen[:MAX_CITATIONS]
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
    
    # Store formatted response data in state for API layer
    state["formatted_response"] = {
        "content": content,
        "finish_reason": finish_reason,
        "citations": [c.model_dump() for c in citations],
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

    return state