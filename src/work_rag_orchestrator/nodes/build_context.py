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
    
    # Build system message — Persian, helpful, professional, grounded, v7-optimized
    system_message = (
        "شما دستیار هوشمند و حرفه‌ای اعتبارسنجی ایران (ICS) هستید — لحنی دوستانه، محترمانه و دقیق دارید. "
        "پاسخ را بر اساس بخش [Context from Knowledge Base] به صورت خلاصه و مفید بنویسید؛ متن‌ها را کپی نکنید، بلکه آن‌ها را با زبان خودتان بازنویسی و ترکیب کنید و نکتهٔ اصلی را واضح توضیح دهید. "
        "هر گزارهٔ کلیدی را با ارجاع شماره‌دار [1]، [2]... پشتیبانی کنید، اما ارجاع را طبیعی در پایان جمله بیاورید. "
        "ساختار پیشنهادی: یک جملهٔ مقدمهٔ کوتاه، سپس ۲-۴ نکتهٔ کلیدی به صورت فهرست یا پاراگراف کوتاه، و در پایان یک جمع‌بندی کاربردی در یک جمله. "
        "اگر پاسخ مستقیم در متن‌ها نیست، از میان نزدیک‌ترین اطلاعات مرتبط یک پاسخ مفید بسازید و تفاوت ظریف را توضیح دهید؛ فقط وقتی هیچ محتوای مرتبطی وجود ندارد بگویید «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» "
        "همیشه فارسی پاسخ دهید — حتی اگر سوال انگلیسی باشد. "
        "برای احوال‌پرسی ساده مانند «سلام» فقط یک پاسخ کوتاه و گرم بدهید بدون ارجاع (مثلاً «سلام! خوش آمدید — چطور می‌توانم در امور اعتباری راهنمایی‌تان کنم؟»). "
        "از تکرار عین جملات طولانی متن‌ها خودداری کنید؛ اطلاعات را خلاصه، روان و کاربردی ارائه دهید."
    )
    
    # Build prompt messages for generation
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"},
    ]
    
    state["prompt_messages"] = prompt_messages
    log.info("Built context with %d sources for request %s", len(chunks), request_id)
    return state