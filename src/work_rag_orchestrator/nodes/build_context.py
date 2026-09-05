"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
from ..state import RAGState

log = logging.getLogger(__name__)

# Max context size — increased for v7 (2077 chunks, more diverse) to improve recall for IVA questions
MAX_CONTEXT_CHARS = 6000
MAX_CHUNKS = 5


async def build_context(state: RAGState) -> RAGState:
    """
    Create a bounded system/context message with numbered source IDs.
    
    Uses retrieved_chunks to build context. Limits total size and chunk count.
    Stores prompt_messages in state for the generation step.
    """
    request_id = state["request_id"]
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    # Limit chunks
    chunks = chunks[:MAX_CHUNKS]
    
    # Build context with numbered sources
    context_parts = ["[Context from Knowledge Base]"]
    
    for i, chunk in enumerate(chunks, 1):
        source_id = f"[{i}]"
        title = chunk.get("title", "Unknown")
        heading = chunk.get("heading", "")
        content = chunk.get("content", "")
        
        context_entry = f"{source_id} Title: {title}"
        if heading:
            context_entry += f"\nHeading: {heading}"
        context_entry += f"\nContent: {content}"
        
        context_parts.append(context_entry)
    
    context_text = "\n\n".join(context_parts)
    
    # Truncate if too long
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    
    # Build system message — Persian, helpful, always cite, no English, v7-optimized for 2077 chunks
    system_message = (
        "شما دستیار هوشمند اعتبارسنجی ایران (ICS) هستید. "
        "فقط بر اساس متن‌های داخل بخش [Context from Knowledge Base] پاسخ دهید. "
        "اگر پاسخ مستقیم در متن‌ها نیست، نزدیک‌ترین اطلاعات مرتبط را از متن‌ها استخراج و خلاصه کنید و با ذکر منابع [1],[2]... پاسخ دهید؛ فقط در صورتی که هیچ اطلاعات مرتبطی در متن‌ها وجود ندارد بگویید «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» "
        "همیشه به زبان فارسی پاسخ دهید — حتی اگر سوال به انگلیسی باشد، هرگز به انگلیسی پاسخ ندهید. "
        "برای احوال‌پرسی ساده مانند «سلام» با لحنی دوستانه و کوتاه به فارسی پاسخ دهید (مثلاً «سلام. چطور می‌توانم به شما کمک کنم؟») و نیازی به ارجاع نیست. "
        "در سایر موارد حتماً منابع را با براکت‌های شماره‌دار [1]، [2] و ... ارجاع دهید و هیچ اطلاعاتی خارج از متن‌ها نسازید. "
        "اگر متن‌ها حاوی اطلاعات ناقص اما مرتبط هستند (مثلاً درباره امتیاز 250 یا گزارش چک)، آن اطلاعات را با ارجاع خلاصه کنید."
    )
    
    # Build prompt messages for generation
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"},
    ]
    
    state["prompt_messages"] = prompt_messages
    log.info("Built context with %d sources for request %s", len(chunks), request_id)
    return state