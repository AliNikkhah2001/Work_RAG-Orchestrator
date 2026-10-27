"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing
from ..rewrite import last_exchanges_text, HISTORY_MAX_CHARS

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
    history_excerpt = last_exchanges_text(
        _strip_notfound_exchanges(messages), include_current=False, max_chars=1000
    )
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
        "از میان حداکثر ۲۰ منبع ارائه‌شده، فقط منابعی را در نظر بگیرید که واقعاً به پرسش پاسخ می‌دهند؛ اگر منبعی فقط از نظر لغوی شبیه سوال است ولی پاسخی در آن نیست، نادیده‌اش بگیرید. "
        "مهم: پاسخ را طوری بنویسید که گویی از دانش خودتان است — طبیعی و انسانی، مثل یک کارشناس که جواب می‌دهد. هیچ‌وقت به منابع، متن‌ها، زمینه (context)، بازیابی (retrieval)، رتبه‌بندی یا برچسب‌هایی مثل CE و RRF و BOTH اشاره نکنید؛ "
        "عبارت‌هایی مثل «در منابع آمده»، «در متن‌ها ذکر شده»، «بر اساس زمینه بازیابی‌شده» یا هر اشارهٔ مشابه به منبع اطلاعات ممنوع است. "
        "اگر هیچ‌یک از منابع پاسخ پرسش را ندارند، فقط در یک جملهٔ کوتاه بگویید «بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد.» و تمام کنید — "
        "اگر در تاریخچهٔ گفت‌وگو پاسخ «یافت نشد» قبلاً داده شده باشد، آن پاسخ منسوخ است و ملاک قرار نمی‌گیرد؛ فقط بر اساس منابع همین پیام دوباره ارزیابی کنید و اگر پاسخ در منابع هست، حتماً پاسخ دهید. "
        "متن را طولانی نکنید و حدس نزنید. "
        "هرگز نصائح مربوط به اشخاص حقیقی (افراد/فردی) را به اشخاص حقوقی (شرکت/سازمان) تعمیم ندهید، و برعکس: اگر منابع فقط دربارهٔ یک نوع موجودیت بحث می‌کنند در حالی که کاربر دربارهٔ نوع دیگر می‌پرسد، بر اساس همان منابع پاسخ دهید؛ اگر منبع صریحاً «خیر» یا «تعریف نشده» می‌گوید، همان را بگویید. ساخت رتبه یا امتیاز چک برای شرکت‌ها از منابع فردی ممنوع است. "
        "مستقیم و دقیق به خودِ سوال پاسخ دهید؛ مقدمهٔ کلیشه‌ای (مثل «موارد زیر را یافتم») ننویسید و پاسخ را در چند پاراگراف منسجم و کاربردی سازمان دهید. "
        "قالب پاسخ متن سادهٔ فارسی است — از هیچ قالب‌بندی مارک‌داون استفاده نکنید: نه ستاره (*)، نه بولد (**)، نه هشتگ (#)، نه خط تیره در ابتدای خط، نه بک‌تیک. "
        "پاسخ را در چند پاراگراف بنویسید و بین پاراگراف‌ها حتماً یک خط خالی بگذارید؛ هر پاراگراف دقیقاً یک موضوع را پوشش دهد و بیش از ۴ جمله نباشد. "
        "ساختار پیشنهادی: پاراگراف اول تعریف، پاراگراف‌های بعد هر کدام یک جنبه (کاربرد، شرایط، مراحل)، هر کدام جدا با خط خالی. "
        "تک‌بلوک متنی بدون خط خالی ممنوع است. "
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

    # --- Inject entity-list tool-call results into the system message ---
    # When the retrieve node fired the supplement tool (reason codes,
    # shareholders, loan titles, ...), render each returned list as an
    # [Entity Data] block so Gemma (which may not render role="tool" input)
    # can consume the matching dictionary rows.
    if state.get("supplemental"):
        try:
            import json as _json
            ent_blocks = []
            # Hard budget: entity rows share the model context with the RAG
            # chunks; one reason-code list alone can be ~100KB. The Triton
            # upstream has an 8192-token window and rejects the request
            # (HTTP 400 max_seq_len overflow) when the total prompt exceeds
            # it — keep the entity section small enough that chunk context
            # + instructions still fit (chars ≈ tokens*3 for Persian).
            _total_budget = 4000
            _used = 0
            for ent in state["supplemental"]:
                content = ent.get("content")
                if content is None:
                    content = {}
                data_json = _json.dumps(content, ensure_ascii=False)
                if _used + len(data_json) > _total_budget:
                    _room = _total_budget - _used
                    if _room < 200:
                        break
                    data_json = data_json[:_room] + "…(truncated)"
                _used += len(data_json)
                ent_blocks.append(
                    f"List: {ent.get('list_name')}\n"
                    f"Description: {ent.get('description','')}\n"
                    f"Data: {data_json}"
                )
            if not ent_blocks:
                raise RuntimeError("no room for entity data")
            entity_section = (
                "\n\n[Entity Data from supplement tool]\n\n"
                + "\n\n---\n\n".join(ent_blocks)
                + "\n\nاگر در بخش [Entity Data] اطلاعاتی مرتبط با پرسش کاربر وجود دارد "
                "(مثل کد دلیل کاهش امتیاز، نام سهامدار، عنوان وام یا نرخ سود)، از آن‌ها برای پاسخ به پرسش استفاده کنید "
                "و جزئیات مربوطه را ذکر کنید."
            )
            system_message += entity_section
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