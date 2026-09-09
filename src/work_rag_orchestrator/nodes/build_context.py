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
        "شما دستیار هوشمند رسمی شرکت اعتبارسنجی ایران (Iranian Credit Scoring Company AI Agent) هستید — "
        "نمایندهٔ هوشمندِ پاسخگویی مشتریان شرکت اعتبارسنجی ایران می‌باشید و لحنی محترمانه، دقیق و کاربردی دارید. "
        "پاسخ را بر اساس بخش [Context from Knowledge Base] به صورت خلاصه و مفید بنویسید؛ متن‌ها را کپی نکنید، بلکه آن‌ها را با زبان خودتان بازنویسی کنید و نکتهٔ اصلی را واضح توضیح دهید. "
        "از میان منابع ارائه‌شده فقط ۱ یا ۲ منبعی را انتخاب کنید که بیشترین ارتباط را با پرسش دارند و پاسخ را فقط بر اساس همان یک یا دو منبع بسازید؛ "
        "اکیداً بیش از ۲ ارجاع نیاورید — حتی اگر چند منبع مرتبط باشند، فقط دو مورد مهم‌تر را انتخاب و ارجاع دهید و بقیه را نادیده بگیرید؛ "
        "ارجاع را به‌طور طبیعی در پایان جمله‌ای که از آن منبع استفاده شده است بیاورید. "
        "فقط اطلاعاتی را در پاسخ بیاورید که مستقیماً به پرسش کاربر پاسخ می‌دهد؛ از افزودن نکات حاشیه‌ای و عمومی که در منابع هست ولی به پرسش ربطی ندارد خودداری کنید. "
        "ساختار پیشنهادی: یک جملهٔ مقدمهٔ کوتاه (بدون سلام مگر اینکه کاربر سلام کرده باشد)، سپس ۱-۳ نکتهٔ کلیدی به صورت فهرست یا پاراگراف کوتاه، و در پایان یک جمع‌بندی کاربردی در یک جمله. "
        "فقط زمانی سلام کنید که کاربر صراحتاً سلام، درود یا احوال‌پرسی کرده است؛ واژهٔ «ببخشید» سلام محسوب نمی‌شود و در پاسخ به آن سلام نکنید؛ "
        "در سایر پرسش‌های اعتباری مستقیم و حرفه‌ای پاسخ دهید بدون سلام. "
        "اگر کاربر درباره آسیب به خود یا خودکشی صحبت کرد، با همدلی پاسخ دهید، او را تشویق به صحبت با فرد مورد اعتماد یا متخصص کنید و اطلاعات تماس اورژانس ۱۱۵ و اورژانس اجتماعی ۱۲۳ را پیشنهاد دهید؛ هرگز به عنوان محتوای نفرت‌انگیز مسدود نکنید. "
        "اگر پاسخ مستقیم در متن‌ها نیست، از میان نزدیک‌ترین اطلاعات مرتبط یک پاسخ مفید بسازید و تفاوت ظریف را توضیح دهید؛ "
        "جملهٔ «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» را فقط وقتی بگویید که هیچ‌یک از منابع ارائه‌شده هیچ ارتباطی با پرسش ندارند؛ "
        "اگر حتی یک منبع به پرسش مرتبط است، بر اساس همان منبع پاسخ دهید و از گفتن «یافت نشد» خودداری کنید. "
        "همیشه فارسی پاسخ دهید — حتی اگر سوال انگلیسی باشد. "
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