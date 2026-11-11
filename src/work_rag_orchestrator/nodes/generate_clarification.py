"""Generate clarifying question when retrieval is scattered."""
import logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings
from ..content_registry import require_prompt

log = logging.getLogger(__name__)


async def generate_clarification(state: RAGState) -> RAGState:
    request_id = state["request_id"]
    query = state.get("query", "")
    from ..content_registry import clarify_option_labels
    available = state.get("available_subspaces") or clarify_option_labels()
    # Colour bands (آبی/سبز/…) are cheque-report UI — ask چک vs تسهیلات, not حقیقی/حقوقی.
    text = ""
    try:
        from ..subspace_routing import query_has_cheque_color_rank

        if query_has_cheque_color_rank(query):
            text = (
                "آیا منظورتان رتبه رنگی گزارش اعتبارسنجی چک است "
                "(مثل سبز/آبی) یا رتبه تسهیلات؟"
            )
    except Exception:
        pass
    if not text:
        prompt = require_prompt("clarify", query=query)
        try:
            settings = get_settings()
            async with GuardrailsClient() as client:
                req = GuardrailsChatRequest(
                    model=settings.upstream_llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.2,
                )
                resp = await client.chat_completion(req, request_id)
                text = resp.choices[0].message.get("content", "").strip() if resp.choices else ""
                if not text:
                    text = require_prompt(
                        "clarify_fallback",
                        options="، ".join(str(x) for x in available),
                    )
        except Exception as e:
            log.warning("Clarification generation failed: %s", e)
            text = require_prompt(
                "clarify_fallback",
                options="، ".join(str(x) for x in available),
            )
    state["clarifying_question"] = text
    state["needs_clarification"] = True
    try:
        from ..tracing import trace_span
        trace_span(request_id, "generate_clarification", span_input=query, span_output=text)
    except Exception:
        pass
    state["answer"] = text
    state["formatted_response"] = {"content": text, "finish_reason": "clarify", "citations": []}
    return state
