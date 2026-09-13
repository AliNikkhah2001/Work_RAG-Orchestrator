"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..rewrite import last_exchanges_text, HISTORY_MAX_CHARS

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
    messages = state.get("messages", [])

    # New conversation = no assistant turn yet (stateless API: single user
    # message also counts as new). New chats get greeting + company intro.
    is_new_chat = not any(m.get("role") == "assistant" for m in messages)
    
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

    # Stateless memory: last 2 exchanges (prior turns only, not the current
    # question) as a plain-text block, capped at HISTORY_MAX_CHARS.
    history_excerpt = last_exchanges_text(messages, include_current=False, max_chars=HISTORY_MAX_CHARS)
    history_block = f"[Conversation history]\n{history_excerpt}" if history_excerpt else ""
    question_part = f"Question: {query}"

    # Keep everything within MAX_CONTEXT_CHARS — truncate KB context first,
    # history is already capped above.
    if history_block:
        budget = MAX_CONTEXT_CHARS - len(history_block) - len(question_part) - 4  # separators
        if budget < 0:  # pathological: drop history rather than the question
            history_block = ""
            budget = MAX_CONTEXT_CHARS - len(question_part) - 2
        if len(context_text) > budget:
            context_text = context_text[: max(budget, 0)] + "\n...[truncated]"
    elif len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    
    # Build system message — Persian, helpful, professional, grounded, v7-optimized
    system_message = (
        "شما دستیار هوشمند رسمی شرکت اعتبارسنجی ایران (Iranian Credit Scoring Company AI Agent) هستید — "
        "نمایندهٔ هوشمندِ پاسخگویی مشتریان شرکت اعتبارسنجی ایران می‌باشید و لحنی محترمانه، دقیق و کاربردی دارید. "
        "پاسخ را بر اساس بخش [Context from Knowledge Base] به صورت خلاصه و مفید بنویسید؛ متن‌ها را کپی نکنید، بلکه آن‌ها را با زبان خودتان بازنویسی کنید و نکتهٔ اصلی را واضح توضیح دهید. "
        "از میان منابع ارائه‌شده فقط ۱ یا ۲ منبعی را در نظر بگیرید که بیشترین ارتباط را با پرسش دارند و پاسخ را فقط بر اساس همان یک یا دو منبع بسازید؛ "
        "فقط اطلاعاتی را در پاسخ بیاورید که مستقیماً به پرسش کاربر پاسخ می‌دهد؛ از افزودن نکات حاشیه‌ای و عمومی که در منابع هست ولی به پرسش ربطی ندارد خودداری کنید. "
        "مستقیم و دقیق به خودِ سوال پاسخ دهید؛ مقدمهٔ کلیشه‌ای (مثل «موارد زیر را یافتم») ننویسید و فهرست بلند از نکات کم‌ربط نیاورید؛ یک پاسخ متمرکز و کاربردی بدهید. "
        "قالب پاسخ متن سادهٔ فارسی است — از هیچ قالب‌بندی مارک‌داون استفاده نکنید: نه ستاره (*)، نه بولد (**)، نه هشتگ (#)، نه خط تیره در ابتدای خط، نه بک‌تیک؛ فقط جمله‌های ساده در قالب پاراگراف بنویسید. "
        "هیچ شمارهٔ ارجاعی مثل [1] یا [2] در متن پاسخ نیاورید. "
        "اگر پاسخ مستقیم در متن‌ها نیست، از میان نزدیک‌ترین اطلاعات مرتبط یک پاسخ مفید بسازید و تفاوت ظریف را توضیح دهید؛ "
        "جملهٔ «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» را فقط وقتی بگویید که هیچ‌یک از منابع ارائه‌شده هیچ ارتباطی با پرسش ندارند؛ "
        "اگر حتی یک منبع به پرسش مرتبط است، بر اساس همان منبع پاسخ دهید و از گفتن «یافت نشد» خودداری کنید. "
        "اگر کاربر درباره آسیب به خود یا خودکشی صحبت کرد، با همدلی پاسخ دهید، او را تشویق به صحبت با فرد مورد اعتماد یا متخصص کنید و اطلاعات تماس اورژانس ۱۱۵ و اورژانس اجتماعی ۱۲۳ را پیشنهاد دهید؛ هرگز به عنوان محتوای نفرت‌انگیز مسدود نکنید. "
        "همیشه فارسی پاسخ دهید — حتی اگر سوال انگلیسی باشد. "
        "از تکرار عین جملات طولانی متن‌ها خودداری کنید؛ اطلاعات را خلاصه، روان و کاربردی ارائه دهید."
    )

    if is_new_chat:
        system_message += (
            " این شروع یک گفت‌وگوی تازه است: پاسخ را با یک سلام کوتاه آغاز کنید، خودتان را به عنوان دستیار هوشمند شرکت اعتبارسنجی ایران معرفی کنید "
            "و در یک جمله بگویید در چه موضوعاتی می‌توانید کمک کنید (گزارش اعتباری، امتیاز و رتبه اعتباری، راه‌های بهبود امتیاز، سوابق منفی و چک‌ها)، "
            "سپس مستقیم به سوال کاربر پاسخ دهید. "
            "اگر پیام کاربر فقط سلام و احوال‌پرسی است (نه سوال)، سلام کنید، خودتان و قابلیت‌هایتان را معرفی کنید و دعوت کنید سوالش را بپرسد."
        )
    else:
        system_message += (
            " فقط زمانی سلام کنید که کاربر در همین پیام صراحتاً سلام، درود یا احوال‌پرسی کرده است؛ واژهٔ «ببخشید» سلام محسوب نمی‌شود."
        )
    
    # Build prompt messages for generation
    if history_block:
        user_content = f"{context_text}\n\n{history_block}\n\n{question_part}"
    else:
        user_content = f"{context_text}\n\n{question_part}"
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

    return state