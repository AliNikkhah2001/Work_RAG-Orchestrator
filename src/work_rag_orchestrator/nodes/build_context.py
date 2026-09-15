"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..rewrite import last_exchanges_text, HISTORY_MAX_CHARS

log = logging.getLogger(__name__)

# Max context size — 20-chunk generation upgrade: wider recall window for
# hybrid (RRF + CE rerank) retrieval; KB context capped at 9000 chars.
MAX_CONTEXT_CHARS = 9000
MAX_CHUNKS = 20


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
    
    # Greeting-only: brief reply, no KB context, no history, no citations.
    greeting_kind = state.get("greeting_only")
    if greeting_kind:
        if greeting_kind == "farewell":
            greeting_system = (
                "کاربر دارد خداحافظی می‌کند. فقط در یک جملهٔ کوتاه فارسی خداحافظی کن "
                "و بگو هر وقت سوالی دربارهٔ گزارش یا امتیاز اعتباری داشت در خدمتی. "
                "هیچ منبع و ارجاعی نیاور و متن طولانی ننویس."
            )
        elif greeting_kind == "thanks":
            greeting_system = (
                "کاربر دارد تشکر می‌کند. فقط در یک جملهٔ کوتاه فارسی پاسخ مودبانه بده "
                "(مثلاً خواهش می‌کنم) و بگو اگر سوال دیگری دربارهٔ گزارش یا امتیاز اعتباری "
                "داشت در خدمتی. هیچ منبع و ارجاعی نیاور و متن طولانی ننویس."
            )
        else:
            greeting_system = (
                "کاربر فقط احوال‌پرسی کرده و سوالی نپرسیده است. فقط در دو جملهٔ کوتاه فارسی "
                "جواب بده: سلام کن، خودت را دستیار هوشمند شرکت اعتبارسنجی ایران معرفی کن و "
                "دعوت کن سوالش را دربارهٔ گزارش اعتباری، امتیاز و رتبه، یا سوابق و چک‌ها بپرسد. "
                "هیچ منبعی نداری، هیچ ارجاعی نیاور، متن طولانی ننویس و چیزی از خودت اضافه نکن."
            )
        state["prompt_messages"] = [
            {"role": "system", "content": greeting_system},
            {"role": "user", "content": query},
        ]
        log.info("Built greeting-only context (%s) for request %s", greeting_kind, request_id)
        return state

    # Limit chunks (retrieval agent guarantees up to 20 items with
    # chunk_id/document_id/title/heading/content/score/source/rank_rrf/
    # rank_ce/hybrid_score/rerank_score keys; be defensive anyway).
    chunks = chunks[:MAX_CHUNKS]

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

    # Stateless memory: last 2 exchanges (prior turns only, not the current
    # question) as a plain-text block, capped at 1000 chars.
    history_excerpt = last_exchanges_text(messages, include_current=False, max_chars=1000)
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
    
    # Build system message — Persian, helpful, professional, grounded.
    # Long/detailed generation: multi-paragraph answers covering all relevant
    # aspects from the (up to 20) retrieved sources.
    system_message = (
        "شما دستیار هوشمند رسمی شرکت اعتبارسنجی ایران (Iranian Credit Scoring Company AI Agent) هستید — "
        "نمایندهٔ هوشمندِ پاسخگویی مشتریان شرکت اعتبارسنجی ایران می‌باشید و لحنی محترمانه، دقیق و کاربردی دارید. "
        "پاسخ را بر اساس بخش [Context from Knowledge Base] به صورت طولانی، مفصل و چندپاراگرافی بنویسید؛ همهٔ جنبه‌های مرتبط با پرسش را که در منابع آمده پوشش دهید و متن‌ها را کپی نکنید، بلکه آن‌ها را با زبان خودتان بازنویسی کنید. "
        "از میان حداکثر ۲۰ منبع ارائه‌شده، فقط منابعی را در نظر بگیرید که واقعاً به پرسش پاسخ می‌دهند؛ اگر منبعی فقط از نظر لغوی شبیه سوال است ولی پاسخی در آن نیست، نادیده‌اش بگیرید و به آن استناد نکنید. "
        "فقط برای نکته‌هایی که واقعاً از منابع استفاده کرده‌اید، با زبان طبیعی مشخص کنید از کدام دسته آمده است؛ برای منابعی با برچسب CE بنویسید «در منابع رتبه‌بندی معنایی آمده…»، "
        "برای منابعی با برچسب RRF بنویسید «در منابع رتبه‌بندی اولیه آمده…» و برای منابعی با برچسب BOTH بنویسید «در منابعی که هر دو روش بازیابی تأیید کرده‌اند آمده…». "
        "اگر هیچ‌یک از منابع پاسخ پرسش را ندارند، فقط در یک جملهٔ کوتاه بگویید «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» و تمام کنید — "
        "متن را طولانی نکنید، حدس نزنید و به منابع نامرتبط استناد نکنید. "
        "مستقیم و دقیق به خودِ سوال پاسخ دهید؛ مقدمهٔ کلیشه‌ای (مثل «موارد زیر را یافتم») ننویسید و پاسخ را در چند پاراگراف منسجم و کاربردی سازمان دهید. "
        "قالب پاسخ متن سادهٔ فارسی است — از هیچ قالب‌بندی مارک‌داون استفاده نکنید: نه ستاره (*)، نه بولد (**)، نه هشتگ (#)، نه خط تیره در ابتدای خط، نه بک‌تیک؛ فقط جمله‌های ساده در قالب پاراگراف بنویسید. "
        "هیچ شمارهٔ ارجاعی مثل [1] یا [2] در متن پاسخ نیاورید. "
        "اگر پاسخ مستقیم در متن‌ها نیست ولی دست‌کم یک منبع ارتباط واقعی با پرسش دارد، از میان نزدیک‌ترین اطلاعات مرتبط یک پاسخ مفید بسازید و تفاوت ظریف را توضیح دهید. "
        "اگر کاربر درباره آسیب به خود یا خودکشی صحبت کرد، با همدلی پاسخ دهید، او را تشویق به صحبت با فرد مورد اعتماد یا متخصص کنید و اطلاعات تماس اورژانس ۱۱۵ و اورژانس اجتماعی ۱۲۳ را پیشنهاد دهید؛ هرگز به عنوان محتوای نفرت‌انگیز مسدود نکنید. "
        "همیشه فارسی پاسخ دهید — حتی اگر سوال انگلیسی باشد. "
        "از تکرار عین جملات طولانی متن‌ها خودداری کنید؛ اطلاعات را روان، دقیق و کاربردی ارائه دهید."
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