"""Generate clarifying question when retrieval is scattered."""
import logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings

log = logging.getLogger(__name__)

CLARIFY_PROMPT = '''شما دستیار هوشمند شرکت اعتبارسنجی ایران هستید.
کاربر سوالی مبهم پرسیده که می‌تواند به چند بخش از پایگاه دانش مربوط باشد:
- تسهیلات افراد (اعتبارسنجی تسهیلات برای اشخاص حقیقی)
- گزارش چک (امتیاز چک، برگشتی)
- تسهیلات شرکت‌ها (اشخاص حقوقی)
- عمومی (معرفی، کاربردها)

سوال کاربر: {query}

یک سوال مشخص‌کننده کوتاه و مودبانه به فارسی بپرسید که کاربر مشخص کند کدام بخش مدنظر است.
مثال: آیا منظورتان امتیاز چک است یا امتیاز تسهیلات؟
فقط یک سوال بپرسید.
'''

async def generate_clarification(state: RAGState) -> RAGState:
    request_id = state["request_id"]
    query = state.get("query", "")
    available = state.get("available_subspaces") or ["تسهیلات افراد", "گزارش چک", "تسهیلات شرکت‌ها"]
    prompt = CLARIFY_PROMPT.format(query=query)
    try:
        settings = get_settings()
        async with GuardrailsClient() as client:
            req = GuardrailsChatRequest(model=settings.upstream_llm_model, messages=[{"role": "user", "content": prompt}], max_tokens=256, temperature=0.2)
            resp = await client.chat_completion(req, request_id)
            text = resp.choices[0].message.get("content", "").strip() if resp.choices else ""
            if not text:
                text = "لطفاً مشخص کنید منظورتان کدام بخش است: تسهیلات افراد، گزارش چک، یا تسهیلات شرکت‌ها؟"
    except Exception as e:
        log.warning("Clarification generation failed: %s", e)
        text = "لطفاً مشخص کنید منظورتان کدام بخش است: تسهیلات افراد، گزارش چک، یا تسهیلات شرکت‌ها؟"
    state["clarifying_question"] = text
    state["needs_clarification"] = True
    # Set formatted_response so graph can END with clarify
    try:
        from ..tracing import trace_span
        trace_span(request_id, "generate_clarification", span_input=query, span_output=text)
    except Exception:
        pass
    # Also set answer for format_response to use if needed
    state["answer"] = text
    state["formatted_response"] = {"content": text, "finish_reason": "clarify", "citations": []}
    return state
