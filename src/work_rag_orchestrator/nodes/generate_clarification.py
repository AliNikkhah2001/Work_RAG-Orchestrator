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
    # A reason code defined in >1 model has a fixed, model-specific question;
    # build it from the membership so it survives later state rewrites.
    _rc_mem = state.get("reason_code_membership") or []
    _rc_models = {m.get("model") for m in _rc_mem if isinstance(m, dict) and m.get("model")}
    if len(_rc_models) > 1 and not state.get("reason_code_model"):
        _labels: list[str] = []
        for m in _rc_mem:
            lbl = m.get("model_label") or m.get("model")
            if lbl and lbl not in _labels:
                _labels.append(str(lbl))
        text = (
            "این کد در چند مدل وجود دارد: "
            + "، ".join(_labels)
            + ". کدام مدل مدنظر شماست؟"
        )
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
    # Buttons must match the question. Derive the option keys from the final text
    # (same mapping magic_emit uses) and store them, so cards and text share a source.
    try:
        from ..magic_emit import _clarify_items, _CLARIFY_CARDS

        keyed = {id(v): k for k, v in _CLARIFY_CARDS.items()}
        cards = _clarify_items({**state, "clarifying_question": text})
        state["clarify_options"] = [keyed[id(c)] for c in cards if id(c) in keyed]
    except Exception as e:
        log.warning("clarify option derivation failed: %s", e)
    try:
        from ..tracing import trace_span
        trace_span(request_id, "generate_clarification", span_input=query, span_output=text)
    except Exception:
        pass
    # Post-hoc clarify cards — kept off the message body (rag.magic).
    try:
        from ..magic_emit import append_magic_tokens

        text = append_magic_tokens(text, state, finish_reason="clarify")
    except Exception as e:
        log.warning("clarify magic_emit failed: %s", e)
    magic_payload = state.get("magic_payload") or []
    state["answer"] = text
    state["clarifying_question"] = text
    state["formatted_response"] = {
        "content": text,
        "finish_reason": "clarify",
        "citations": [],
        "magic": magic_payload if magic_payload else None,
    }
    return state
